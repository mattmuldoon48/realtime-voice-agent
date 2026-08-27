"""Unit tests for deterministic DynamoDB persistence mapping and conditions."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from botocore.exceptions import ClientError  # type: ignore[import-untyped]

from realtime_voice_agent.config import PersistenceRuntimeConfig
from realtime_voice_agent.persistence.dynamodb import DynamoPersistenceStore
from realtime_voice_agent.persistence.errors import PersistenceError
from realtime_voice_agent.persistence.models import TranscriptTurn
from realtime_voice_agent.transcript import TranscriptSpeaker


class FakeDynamoClient:
    def __init__(self) -> None:
        self.get_items: dict[str, dict[str, object]] = {}
        self.scan_items: list[dict[str, object]] = []
        self.transcript_items: list[dict[str, object]] = []
        self.put_requests: list[dict[str, object]] = []
        self.transaction_requests: list[list[dict[str, object]]] = []
        self.reject_put = False
        self.put_error_code: str | None = None

    def get_item(self, **request: object) -> dict[str, object]:
        table_name = str(request["TableName"])
        lookup_key = table_name
        key = request.get("Key")
        if isinstance(key, dict):
            persona_id = key.get("persona_id")
            if isinstance(persona_id, dict) and isinstance(persona_id.get("S"), str):
                lookup_key = f"{table_name}:{persona_id['S']}"
        item = self.get_items.get(lookup_key, self.get_items.get(table_name))
        return {"Item": item} if item is not None else {}

    def scan(self, **request: object) -> dict[str, object]:
        return {"Items": self.scan_items}

    def query(self, **request: object) -> dict[str, object]:
        return {"Items": self.transcript_items}

    def put_item(self, **request: object) -> dict[str, object]:
        error_code = "ConditionalCheckFailedException" if self.reject_put else self.put_error_code
        if error_code is not None:
            raise ClientError(
                {"Error": {"Code": error_code}},
                "PutItem",
            )
        self.put_requests.append(dict(request))
        return {}

    def transact_write_items(self, **request: object) -> dict[str, object]:
        items = request["TransactItems"]
        if not isinstance(items, list):
            raise ValueError("TransactItems must be a list")
        self.transaction_requests.append(items)
        return {}


class FakeWaiter:
    def __init__(self, waits: list[str]) -> None:
        self._waits = waits

    def wait(self, *, TableName: str) -> None:
        self._waits.append(TableName)


class FakeBootstrapClient:
    def __init__(self) -> None:
        self.create_requests: list[dict[str, object]] = []
        self.waits: list[str] = []
        self.ttl_tables: list[str] = []

    def describe_table(self, *, TableName: str) -> dict[str, object]:
        raise ClientError(
            {"Error": {"Code": "ResourceNotFoundException"}},
            "DescribeTable",
        )

    def create_table(self, **request: object) -> dict[str, object]:
        self.create_requests.append(dict(request))
        return {}

    def get_waiter(self, name: str) -> FakeWaiter:
        if name != "table_exists":
            raise ValueError("unexpected waiter")
        return FakeWaiter(self.waits)

    def describe_time_to_live(self, *, TableName: str) -> dict[str, object]:
        return {"TimeToLiveDescription": {"TimeToLiveStatus": "DISABLED"}}

    def update_time_to_live(self, **request: object) -> dict[str, object]:
        self.ttl_tables.append(str(request["TableName"]))
        return {}


def test_table_bootstrap_creates_three_on_demand_tables_and_retention_ttl() -> None:
    client = FakeBootstrapClient()
    store = DynamoPersistenceStore(_CONFIG, client=client)

    store.ensure_tables()

    assert [request["TableName"] for request in client.create_requests] == [
        "personas",
        "sessions",
        "turns",
    ]
    assert all(request["BillingMode"] == "PAY_PER_REQUEST" for request in client.create_requests)
    sessions = client.create_requests[1]
    indexes = sessions["GlobalSecondaryIndexes"]
    assert isinstance(indexes, list)
    assert indexes[0]["IndexName"] == "CallSidIndex"
    assert client.waits == ["personas", "sessions", "turns"]
    assert client.ttl_tables == ["sessions", "turns"]


_CONFIG = PersistenceRuntimeConfig(
    region="us-east-1",
    profile=None,
    personas_table="personas",
    sessions_table="sessions",
    transcripts_table="turns",
    transcript_retention_days=7,
    store_phone_numbers=False,
    queue_max_events=100,
    cleanup_timeout_seconds=5.0,
    max_attempts=3,
    retry_base_delay_seconds=0.1,
)


def test_persona_list_filters_active_pointer_and_decodes_integer_versions() -> None:
    client = FakeDynamoClient()
    client.scan_items = [
        {
            "persona_id": {"S": "__active_persona__"},
            "active_persona_id": {"S": "concierge"},
            "active_persona_version": {"N": "4"},
        },
        _persona_item(persona_id="concierge", version=4, active=True),
        _persona_item(persona_id="scheduler", version=2, active=False),
    ]
    store = DynamoPersistenceStore(_CONFIG, client=client)

    personas = store.list_personas()

    assert [persona.persona_id for persona in personas] == ["concierge", "scheduler"]
    assert [persona.version for persona in personas] == [4, 2]
    assert all(isinstance(persona.version, int) for persona in personas)


def test_create_persona_uses_create_only_condition_and_version_one() -> None:
    client = FakeDynamoClient()
    store = DynamoPersistenceStore(_CONFIG, client=client)

    persona = store.put_persona(
        persona_id="concierge",
        name="Concierge",
        system_prompt="Answer briefly.",
        voice_id="matthew",
        expected_version=0,
    )

    assert persona.version == 1
    assert persona.active is False
    assert len(client.put_requests) == 1
    request = client.put_requests[0]
    assert request["ConditionExpression"] == "attribute_not_exists(persona_id)"
    item = request["Item"]
    assert isinstance(item, dict)
    assert item["version"] == {"N": "1"}


def test_activation_writes_persona_and_single_active_pointer_atomically() -> None:
    client = FakeDynamoClient()
    client.get_items["personas:concierge"] = _persona_item(
        persona_id="concierge",
        version=1,
        active=False,
    )
    store = DynamoPersistenceStore(_CONFIG, client=client)

    persona = store.activate_persona("concierge", expected_version=1)

    assert persona.active is True
    assert persona.version == 2
    assert len(client.transaction_requests) == 1
    transaction = client.transaction_requests[0]
    assert len(transaction) == 2
    persona_put = transaction[0]["Put"]
    pointer_put = transaction[1]["Put"]
    assert isinstance(persona_put, dict)
    assert isinstance(pointer_put, dict)
    assert persona_put["ConditionExpression"] == "#version = :expected"
    assert pointer_put["ConditionExpression"] == "attribute_not_exists(persona_id)"


def test_duplicate_transcript_key_is_reported_as_idempotent_noop() -> None:
    client = FakeDynamoClient()
    client.reject_put = True
    store = DynamoPersistenceStore(_CONFIG, client=client)
    turn = TranscriptTurn(
        session_id="session-1",
        turn_number=1,
        speaker=TranscriptSpeaker.CALLER,
        text="Hello",
        source_event_id="final-1",
        created_at=datetime(2026, 8, 4, tzinfo=UTC),
        expires_at=1_786_435_200,
    )

    assert store.append_transcript_turn(turn) is False


@pytest.mark.parametrize(
    "error_code",
    [
        "ProvisionedThroughputExceededException",
        "ThrottlingException",
        "RequestLimitExceeded",
        "LimitExceededException",
    ],
)
def test_explicit_dynamodb_rejections_are_normalized_as_retryable(error_code: str) -> None:
    client = FakeDynamoClient()
    client.put_error_code = error_code
    store = DynamoPersistenceStore(_CONFIG, client=client)
    turn = TranscriptTurn(
        session_id="session-1",
        turn_number=1,
        speaker=TranscriptSpeaker.CALLER,
        text="Hello",
        source_event_id="final-1",
        created_at=datetime(2026, 8, 4, tzinfo=UTC),
        expires_at=1_786_435_200,
    )

    with pytest.raises(PersistenceError) as captured:
        store.append_transcript_turn(turn)

    assert captured.value.code == error_code
    assert captured.value.retryable is True


def test_transcript_retrieval_returns_numeric_turn_order_and_json_safe_numbers() -> None:
    client = FakeDynamoClient()
    client.get_items["sessions"] = {
        "session_id": {"S": "session-1"},
        "call_sid": {"S": "CA00000000000000000000000000000000"},
        "status": {"S": "COMPLETED"},
        "started_at": {"S": "2026-08-04T12:00:00+00:00"},
        "inbound_media_frames": {"N": "42"},
    }
    client.transcript_items = [
        _turn_item(2, "AGENT", "Welcome", "agent-final"),
        _turn_item(1, "CALLER", "Hello", "caller-final"),
    ]
    store = DynamoPersistenceStore(_CONFIG, client=client)

    transcript = store.get_transcript(session_id="session-1")

    assert transcript.session["inbound_media_frames"] == 42
    assert isinstance(transcript.session["inbound_media_frames"], int)
    assert [turn.turn_number for turn in transcript.turns] == [1, 2]
    assert [turn.text for turn in transcript.turns] == ["Hello", "Welcome"]


def _persona_item(*, persona_id: str, version: int, active: bool) -> dict[str, object]:
    timestamp = "2026-08-04T12:00:00+00:00"
    return {
        "persona_id": {"S": persona_id},
        "name": {"S": persona_id.title()},
        "system_prompt": {"S": "Answer briefly."},
        "voice_id": {"S": "matthew"},
        "version": {"N": str(version)},
        "active": {"BOOL": active},
        "created_at": {"S": timestamp},
        "updated_at": {"S": timestamp},
    }


def _turn_item(
    turn_number: int,
    speaker: str,
    text: str,
    source_event_id: str,
) -> dict[str, object]:
    return {
        "session_id": {"S": "session-1"},
        "turn_number": {"N": str(turn_number)},
        "speaker": {"S": speaker},
        "text": {"S": text},
        "source_event_id": {"S": source_event_id},
        "created_at": {"S": "2026-08-04T12:00:00+00:00"},
        "expires_at": {"N": "1786435200"},
    }
