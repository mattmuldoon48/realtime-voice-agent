"""Unit tests for bounded read-only dependency readiness."""

from __future__ import annotations

import time
from collections.abc import Mapping
from datetime import UTC, datetime

import pytest
from botocore.exceptions import ClientError  # type: ignore[import-untyped]

from realtime_voice_agent.config import ReadinessRuntimeConfig
from realtime_voice_agent.persistence.errors import (
    PersistenceError,
    PersistenceNotFoundError,
)
from realtime_voice_agent.persistence.models import Persona
from realtime_voice_agent.readiness import (
    DependencyReadinessService,
    ReadinessErrorCode,
)


class FakeIdentityClient:
    def __init__(self, *, arn: str = "arn:aws:iam::123456789012:user/readiness") -> None:
        self.arn = arn
        self.error: Exception | None = None

    def get_caller_identity(self, **kwargs: object) -> Mapping[str, object]:
        del kwargs
        if self.error is not None:
            raise self.error
        return {"Arn": self.arn, "Account": "123456789012"}


class SlowIdentityClient(FakeIdentityClient):
    def get_caller_identity(self, **kwargs: object) -> Mapping[str, object]:
        time.sleep(0.05)
        return super().get_caller_identity(**kwargs)


class FakeDynamoReadinessClient:
    def __init__(self) -> None:
        self.tables = _healthy_tables()
        self.ttl = {
            "sessions": {
                "TimeToLiveStatus": "ENABLED",
                "AttributeName": "expires_at",
            },
            "turns": {
                "TimeToLiveStatus": "ENABLED",
                "AttributeName": "expires_at",
            },
        }

    def describe_table(self, **kwargs: object) -> Mapping[str, object]:
        table_name = str(kwargs["TableName"])
        table = self.tables.get(table_name)
        if table is None:
            raise ClientError(
                {"Error": {"Code": "ResourceNotFoundException"}},
                "DescribeTable",
            )
        return {"Table": table}

    def describe_time_to_live(self, **kwargs: object) -> Mapping[str, object]:
        table_name = str(kwargs["TableName"])
        return {"TimeToLiveDescription": self.ttl[table_name]}


class FakePersonaRepository:
    def __init__(self) -> None:
        now = datetime(2026, 8, 5, tzinfo=UTC)
        self.persona = Persona(
            persona_id="concierge",
            name="Concierge",
            system_prompt="Answer briefly.",
            voice_id="matthew",
            version=2,
            active=True,
            created_at=now,
            updated_at=now,
        )
        self.error: Exception | None = None

    def get_active_persona(self) -> Persona:
        if self.error is not None:
            raise self.error
        return self.persona


def _config(*, timeout_seconds: float = 1.0) -> ReadinessRuntimeConfig:
    return ReadinessRuntimeConfig(
        region="us-east-1",
        profile=None,
        personas_table="personas",
        sessions_table="sessions",
        transcripts_table="turns",
        timeout_seconds=timeout_seconds,
    )


def _service(
    *,
    identity: FakeIdentityClient | None = None,
    dynamodb: FakeDynamoReadinessClient | None = None,
    personas: FakePersonaRepository | None = None,
    timeout_seconds: float = 1.0,
) -> DependencyReadinessService:
    return DependencyReadinessService(
        config=_config(timeout_seconds=timeout_seconds),
        identity_client=identity or FakeIdentityClient(),
        dynamodb_client=dynamodb or FakeDynamoReadinessClient(),
        persona_repository=personas or FakePersonaRepository(),
    )


@pytest.mark.asyncio
async def test_readiness_is_healthy_when_every_dependency_contract_matches() -> None:
    result = await _service().check()

    assert result.is_ready is True
    assert result.error_code is None
    assert result.to_document() == {"status": "ready"}


@pytest.mark.asyncio
async def test_readiness_rejects_root_identity() -> None:
    identity = FakeIdentityClient(arn="arn:aws:iam::123456789012:root")

    result = await _service(identity=identity).check()

    assert result.error_code is ReadinessErrorCode.ROOT_AWS_IDENTITY


@pytest.mark.asyncio
async def test_readiness_normalizes_identity_failure() -> None:
    identity = FakeIdentityClient()
    identity.error = RuntimeError("synthetic identity failure")

    result = await _service(identity=identity).check()

    assert result.error_code is ReadinessErrorCode.AWS_IDENTITY_UNAVAILABLE


@pytest.mark.asyncio
async def test_readiness_reports_missing_table() -> None:
    dynamodb = FakeDynamoReadinessClient()
    del dynamodb.tables["personas"]

    result = await _service(dynamodb=dynamodb).check()

    assert result.error_code is ReadinessErrorCode.TABLE_MISSING


@pytest.mark.asyncio
async def test_readiness_reports_inactive_table() -> None:
    dynamodb = FakeDynamoReadinessClient()
    dynamodb.tables["sessions"]["TableStatus"] = "UPDATING"

    result = await _service(dynamodb=dynamodb).check()

    assert result.error_code is ReadinessErrorCode.TABLE_INACTIVE


@pytest.mark.asyncio
async def test_readiness_rejects_incorrect_primary_key_schema() -> None:
    dynamodb = FakeDynamoReadinessClient()
    dynamodb.tables["turns"]["KeySchema"] = [{"AttributeName": "session_id", "KeyType": "HASH"}]

    result = await _service(dynamodb=dynamodb).check()

    assert result.error_code is ReadinessErrorCode.TABLE_KEY_SCHEMA_INVALID


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("indexes", "expected"),
    [
        ([], ReadinessErrorCode.CALL_SID_INDEX_MISSING),
        (
            [
                {
                    "IndexName": "CallSidIndex",
                    "IndexStatus": "CREATING",
                    "KeySchema": [{"AttributeName": "call_sid", "KeyType": "HASH"}],
                }
            ],
            ReadinessErrorCode.CALL_SID_INDEX_INACTIVE,
        ),
    ],
)
async def test_readiness_requires_active_call_sid_index(
    indexes: list[dict[str, object]],
    expected: ReadinessErrorCode,
) -> None:
    dynamodb = FakeDynamoReadinessClient()
    dynamodb.tables["sessions"]["GlobalSecondaryIndexes"] = indexes

    result = await _service(dynamodb=dynamodb).check()

    assert result.error_code is expected


@pytest.mark.asyncio
async def test_readiness_rejects_incorrect_ttl_configuration() -> None:
    dynamodb = FakeDynamoReadinessClient()
    dynamodb.ttl["turns"] = {
        "TimeToLiveStatus": "DISABLED",
        "AttributeName": "wrong_attribute",
    }

    result = await _service(dynamodb=dynamodb).check()

    assert result.error_code is ReadinessErrorCode.TTL_CONFIGURATION_INVALID


@pytest.mark.asyncio
async def test_readiness_reports_missing_active_persona() -> None:
    personas = FakePersonaRepository()
    personas.error = PersistenceNotFoundError("No active persona is configured")

    result = await _service(personas=personas).check()

    assert result.error_code is ReadinessErrorCode.ACTIVE_PERSONA_MISSING


@pytest.mark.asyncio
async def test_readiness_normalizes_active_persona_retrieval_failure() -> None:
    personas = FakePersonaRepository()
    personas.error = PersistenceError("ACTIVE_PERSONA_GET_FAILED:AccessDeniedException")

    result = await _service(personas=personas).check()

    assert result.error_code is ReadinessErrorCode.ACTIVE_PERSONA_UNAVAILABLE
    assert "AccessDenied" not in str(result.to_document())


@pytest.mark.asyncio
async def test_readiness_timeout_returns_bounded_failure() -> None:
    result = await _service(
        identity=SlowIdentityClient(),
        timeout_seconds=0.01,
    ).check()

    assert result.error_code is ReadinessErrorCode.READINESS_TIMEOUT
    assert result.to_document() == {
        "status": "not_ready",
        "error_code": "READINESS_TIMEOUT",
    }


def _healthy_tables() -> dict[str, dict[str, object]]:
    return {
        "personas": {
            "TableStatus": "ACTIVE",
            "KeySchema": [{"AttributeName": "persona_id", "KeyType": "HASH"}],
            "AttributeDefinitions": [{"AttributeName": "persona_id", "AttributeType": "S"}],
        },
        "sessions": {
            "TableStatus": "ACTIVE",
            "KeySchema": [{"AttributeName": "session_id", "KeyType": "HASH"}],
            "AttributeDefinitions": [
                {"AttributeName": "session_id", "AttributeType": "S"},
                {"AttributeName": "call_sid", "AttributeType": "S"},
            ],
            "GlobalSecondaryIndexes": [
                {
                    "IndexName": "CallSidIndex",
                    "IndexStatus": "ACTIVE",
                    "KeySchema": [{"AttributeName": "call_sid", "KeyType": "HASH"}],
                }
            ],
        },
        "turns": {
            "TableStatus": "ACTIVE",
            "KeySchema": [
                {"AttributeName": "session_id", "KeyType": "HASH"},
                {"AttributeName": "turn_number", "KeyType": "RANGE"},
            ],
            "AttributeDefinitions": [
                {"AttributeName": "session_id", "AttributeType": "S"},
                {"AttributeName": "turn_number", "AttributeType": "N"},
            ],
        },
    }
