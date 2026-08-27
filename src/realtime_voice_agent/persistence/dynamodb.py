"""DynamoDB implementation for personas, call sessions, and final transcripts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, Final, cast

import boto3  # type: ignore[import-untyped]
from boto3.dynamodb.types import TypeDeserializer, TypeSerializer  # type: ignore[import-untyped]
from botocore.exceptions import BotoCoreError, ClientError  # type: ignore[import-untyped]

from realtime_voice_agent.config import PersistenceRuntimeConfig
from realtime_voice_agent.persistence.errors import (
    PersistenceConflictError,
    PersistenceError,
    PersistenceNotFoundError,
)
from realtime_voice_agent.persistence.models import (
    Persona,
    SessionStart,
    SessionTerminal,
    TranscriptTurn,
    TranscriptView,
)
from realtime_voice_agent.transcript import TranscriptSpeaker

_ACTIVE_POINTER_ID: Final = "__active_persona__"
_CALL_SID_INDEX: Final = "CallSidIndex"
_TTL_ATTRIBUTE: Final = "expires_at"
_RETRYABLE_REJECTION_CODES: Final = frozenset(
    {
        "ProvisionedThroughputExceededException",
        "ThrottlingException",
        "RequestLimitExceeded",
        "LimitExceededException",
    }
)


class DynamoPersistenceStore:
    """One-process DynamoDB repository using the standard AWS credential chain."""

    def __init__(
        self,
        config: PersistenceRuntimeConfig,
        *,
        client: Any | None = None,
    ) -> None:
        self._config = config
        if client is None:
            try:
                client = boto3.Session(
                    profile_name=config.profile,
                    region_name=config.region,
                ).client("dynamodb")
            except BotoCoreError as error:
                raise PersistenceError("DYNAMODB_CLIENT_CONFIGURATION_FAILED") from error
        self._client: Any = client
        self._serializer = TypeSerializer()
        self._deserializer = TypeDeserializer()

    def ensure_tables(self) -> None:
        """Create the three on-demand tables and TTL settings when absent."""
        self._ensure_table(
            table_name=self._config.personas_table,
            key_schema=[{"AttributeName": "persona_id", "KeyType": "HASH"}],
            attribute_definitions=[{"AttributeName": "persona_id", "AttributeType": "S"}],
        )
        self._ensure_table(
            table_name=self._config.sessions_table,
            key_schema=[{"AttributeName": "session_id", "KeyType": "HASH"}],
            attribute_definitions=[
                {"AttributeName": "session_id", "AttributeType": "S"},
                {"AttributeName": "call_sid", "AttributeType": "S"},
            ],
            global_secondary_indexes=[
                {
                    "IndexName": _CALL_SID_INDEX,
                    "KeySchema": [{"AttributeName": "call_sid", "KeyType": "HASH"}],
                    "Projection": {"ProjectionType": "ALL"},
                }
            ],
        )
        self._ensure_table(
            table_name=self._config.transcripts_table,
            key_schema=[
                {"AttributeName": "session_id", "KeyType": "HASH"},
                {"AttributeName": "turn_number", "KeyType": "RANGE"},
            ],
            attribute_definitions=[
                {"AttributeName": "session_id", "AttributeType": "S"},
                {"AttributeName": "turn_number", "AttributeType": "N"},
            ],
        )
        self._ensure_ttl(self._config.sessions_table)
        self._ensure_ttl(self._config.transcripts_table)

    def list_personas(self) -> Sequence[Persona]:
        """Return all administrable personas in stable ID order."""
        items: list[dict[str, object]] = []
        request: dict[str, object] = {"TableName": self._config.personas_table}
        try:
            while True:
                response = cast(dict[str, object], self._client.scan(**request))
                raw_items = cast(list[Mapping[str, object]], response.get("Items", []))
                items.extend(self._deserialize_item(item) for item in raw_items)
                last_key = response.get("LastEvaluatedKey")
                if not isinstance(last_key, dict):
                    break
                request["ExclusiveStartKey"] = last_key
        except ClientError as error:
            raise _persistence_error("PERSONA_LIST_FAILED", error) from error
        personas = [
            _persona_from_item(item)
            for item in items
            if item.get("persona_id") != _ACTIVE_POINTER_ID
        ]
        return tuple(sorted(personas, key=lambda persona: persona.persona_id))

    def get_persona(self, persona_id: str) -> Persona | None:
        """Load one persona by its stable identifier."""
        item = self._get_item(
            self._config.personas_table,
            {"persona_id": persona_id},
            operation="PERSONA_GET_FAILED",
        )
        return _persona_from_item(item) if item is not None else None

    def get_active_persona(self) -> Persona:
        """Resolve the single active pointer and verify its target version."""
        pointer = self._get_item(
            self._config.personas_table,
            {"persona_id": _ACTIVE_POINTER_ID},
            operation="ACTIVE_PERSONA_GET_FAILED",
        )
        if pointer is None:
            raise PersistenceNotFoundError("No active persona is configured")
        persona_id = _required_str(pointer, "active_persona_id")
        expected_version = _required_int(pointer, "active_persona_version")
        persona = self.get_persona(persona_id)
        if persona is None or not persona.active or persona.version != expected_version:
            raise PersistenceError("ACTIVE_PERSONA_POINTER_INVALID")
        return persona

    def put_persona(
        self,
        *,
        persona_id: str,
        name: str,
        system_prompt: str,
        voice_id: str,
        expected_version: int,
    ) -> Persona:
        """Create or replace one persona under an optimistic version condition."""
        if expected_version < 0:
            raise ValueError("expected persona version must not be negative")
        now = datetime.now(UTC)
        existing = self.get_persona(persona_id)
        if expected_version == 0:
            if existing is not None:
                raise PersistenceConflictError("PERSONA_VERSION_CONFLICT")
            persona = Persona(
                persona_id=persona_id,
                name=name,
                system_prompt=system_prompt,
                voice_id=voice_id,
                version=1,
                active=False,
                created_at=now,
                updated_at=now,
            )
            self._conditional_put_persona(persona, expected_version=0)
            return persona
        if existing is None or existing.version != expected_version:
            raise PersistenceConflictError("PERSONA_VERSION_CONFLICT")
        persona = Persona(
            persona_id=persona_id,
            name=name,
            system_prompt=system_prompt,
            voice_id=voice_id,
            version=expected_version + 1,
            active=existing.active,
            created_at=existing.created_at,
            updated_at=now,
        )
        if existing.active:
            self._replace_active_persona(persona, expected_version=expected_version)
        else:
            self._conditional_put_persona(persona, expected_version=expected_version)
        return persona

    def activate_persona(self, persona_id: str, *, expected_version: int) -> Persona:
        """Atomically move the active pointer under optimistic version checks."""
        target = self.get_persona(persona_id)
        if target is None:
            raise PersistenceNotFoundError("PERSONA_NOT_FOUND")
        if target.version != expected_version:
            raise PersistenceConflictError("PERSONA_VERSION_CONFLICT")
        pointer = self._get_item(
            self._config.personas_table,
            {"persona_id": _ACTIVE_POINTER_ID},
            operation="ACTIVE_PERSONA_GET_FAILED",
        )
        previous: Persona | None = None
        if pointer is not None:
            previous_id = _required_str(pointer, "active_persona_id")
            previous = self.get_persona(previous_id)
            if previous is None:
                raise PersistenceError("ACTIVE_PERSONA_POINTER_INVALID")
        activated = Persona(
            persona_id=target.persona_id,
            name=target.name,
            system_prompt=target.system_prompt,
            voice_id=target.voice_id,
            version=target.version + 1,
            active=True,
            created_at=target.created_at,
            updated_at=datetime.now(UTC),
        )
        transaction = [
            self._persona_put_transaction(activated, expected_version=target.version),
            self._pointer_put_transaction(
                activated,
                previous=previous,
                pointer_exists=pointer is not None,
            ),
        ]
        if previous is not None and previous.persona_id != target.persona_id:
            deactivated = Persona(
                persona_id=previous.persona_id,
                name=previous.name,
                system_prompt=previous.system_prompt,
                voice_id=previous.voice_id,
                version=previous.version + 1,
                active=False,
                created_at=previous.created_at,
                updated_at=activated.updated_at,
            )
            transaction.append(
                self._persona_put_transaction(
                    deactivated,
                    expected_version=previous.version,
                )
            )
        try:
            self._client.transact_write_items(TransactItems=transaction)
        except ClientError as error:
            if _error_code(error) == "TransactionCanceledException":
                raise PersistenceConflictError("PERSONA_ACTIVATION_CONFLICT") from error
            raise _persistence_error("PERSONA_ACTIVATION_FAILED", error) from error
        return activated

    def create_session(self, session: SessionStart) -> None:
        """Write one STARTING session without overwriting an existing ID."""
        item: dict[str, object] = {
            "session_id": session.session_id,
            "call_sid": session.call_sid,
            "stream_sid": session.stream_sid,
            "status": "STARTING",
            "persona_id": session.persona.persona_id,
            "persona_name": session.persona.name,
            "persona_version": session.persona.version,
            "persona_system_prompt": session.persona.system_prompt,
            "persona_voice_id": session.persona.voice_id,
            "model_id": session.model_id,
            "started_at": session.started_at.isoformat(),
            "expires_at": session.expires_at,
        }
        try:
            self._client.put_item(
                TableName=self._config.sessions_table,
                Item=self._serialize_item(item),
                ConditionExpression="attribute_not_exists(session_id)",
            )
        except ClientError as error:
            if _error_code(error) == "ConditionalCheckFailedException":
                raise PersistenceConflictError("SESSION_ALREADY_EXISTS") from error
            raise _persistence_error("SESSION_CREATE_FAILED", error) from error

    def mark_session_active(self, session_id: str, activated_at: str) -> None:
        """Transition a persisted STARTING session to ACTIVE exactly once."""
        try:
            self._client.update_item(
                TableName=self._config.sessions_table,
                Key=self._serialize_item({"session_id": session_id}),
                UpdateExpression="SET #status = :active, activated_at = :activated",
                ConditionExpression="#status = :starting",
                ExpressionAttributeNames={"#status": "status"},
                ExpressionAttributeValues=self._serialize_item(
                    {
                        ":active": "ACTIVE",
                        ":starting": "STARTING",
                        ":activated": activated_at,
                    }
                ),
            )
        except ClientError as error:
            if _error_code(error) == "ConditionalCheckFailedException":
                raise PersistenceConflictError("SESSION_ACTIVE_CONFLICT") from error
            raise _persistence_error("SESSION_ACTIVE_FAILED", error) from error

    def append_transcript_turn(self, turn: TranscriptTurn) -> bool:
        """Persist one ordered final turn; return false for an exact duplicate key."""
        item: dict[str, object] = {
            "session_id": turn.session_id,
            "turn_number": turn.turn_number,
            "speaker": turn.speaker.value,
            "text": turn.text,
            "source_event_id": turn.source_event_id,
            "created_at": turn.created_at.isoformat(),
            "is_final": True,
            "expires_at": turn.expires_at,
        }
        try:
            self._client.put_item(
                TableName=self._config.transcripts_table,
                Item=self._serialize_item(item),
                ConditionExpression=(
                    "attribute_not_exists(session_id) AND attribute_not_exists(turn_number)"
                ),
            )
        except ClientError as error:
            if _error_code(error) == "ConditionalCheckFailedException":
                return False
            raise _persistence_error("TRANSCRIPT_WRITE_FAILED", error) from error
        return True

    def finish_session(self, terminal: SessionTerminal) -> bool:
        """Write terminal metadata once; later racing terminal writes are ignored."""
        values: dict[str, object] = {
            ":status": terminal.status.value,
            ":outcome": terminal.outcome,
            ":reason": terminal.termination_reason,
            ":ended": terminal.ended_at.isoformat(),
            ":duration": terminal.duration_ms,
            ":inbound": terminal.inbound_media_frames,
            ":nova_frames": terminal.nova_input_frames,
            ":nova_bytes": terminal.nova_input_pcm16_bytes,
            ":nova_audio": terminal.nova_response_audio_events,
            ":outbound": terminal.outbound_media_frames,
            ":outbound_bytes": terminal.outbound_mulaw_bytes,
            ":caller_turns": terminal.caller_turns,
            ":agent_turns": terminal.agent_turns,
            ":starting": "STARTING",
            ":active": "ACTIVE",
        }
        assignments = [
            "#status = :status",
            "outcome = :outcome",
            "termination_reason = :reason",
            "ended_at = :ended",
            "duration_ms = :duration",
            "inbound_media_frames = :inbound",
            "nova_input_frames = :nova_frames",
            "nova_input_pcm16_bytes = :nova_bytes",
            "nova_response_audio_events = :nova_audio",
            "outbound_media_frames = :outbound",
            "outbound_mulaw_bytes = :outbound_bytes",
            "caller_turns = :caller_turns",
            "agent_turns = :agent_turns",
        ]
        if terminal.failure_code is not None:
            assignments.append("failure_code = :failure")
            values[":failure"] = terminal.failure_code
        if terminal.cleanup_error_code is not None:
            assignments.append("cleanup_error_code = :cleanup")
            values[":cleanup"] = terminal.cleanup_error_code
        try:
            self._client.update_item(
                TableName=self._config.sessions_table,
                Key=self._serialize_item({"session_id": terminal.session_id}),
                UpdateExpression="SET " + ", ".join(assignments),
                ConditionExpression=(
                    "attribute_not_exists(ended_at) AND (#status = :starting OR #status = :active)"
                ),
                ExpressionAttributeNames={"#status": "status"},
                ExpressionAttributeValues=self._serialize_item(values),
            )
        except ClientError as error:
            if _error_code(error) == "ConditionalCheckFailedException":
                return False
            raise _persistence_error("SESSION_TERMINAL_FAILED", error) from error
        return True

    def get_transcript(
        self,
        *,
        session_id: str | None = None,
        call_sid: str | None = None,
    ) -> TranscriptView:
        """Retrieve session metadata and final turns by session ID or Twilio call SID."""
        if (session_id is None) == (call_sid is None):
            raise ValueError("provide exactly one of session_id or call_sid")
        if session_id is None:
            if call_sid is None:
                raise ValueError("call_sid is required")
            session = self._session_for_call_sid(call_sid)
            session_id = _required_str(session, "session_id")
        else:
            loaded = self._get_item(
                self._config.sessions_table,
                {"session_id": session_id},
                operation="SESSION_GET_FAILED",
            )
            if loaded is None:
                raise PersistenceNotFoundError("SESSION_NOT_FOUND")
            session = loaded
        try:
            response = cast(
                dict[str, object],
                self._client.query(
                    TableName=self._config.transcripts_table,
                    KeyConditionExpression="session_id = :session_id",
                    ExpressionAttributeValues=self._serialize_item({":session_id": session_id}),
                    ScanIndexForward=True,
                    ConsistentRead=True,
                ),
            )
        except ClientError as error:
            raise _persistence_error("TRANSCRIPT_GET_FAILED", error) from error
        raw_items = cast(list[Mapping[str, object]], response.get("Items", []))
        turns = tuple(
            sorted(
                (_transcript_from_item(self._deserialize_item(item)) for item in raw_items),
                key=lambda turn: turn.turn_number,
            )
        )
        return TranscriptView(session=session, turns=turns)

    def _session_for_call_sid(self, call_sid: str) -> dict[str, object]:
        try:
            response = cast(
                dict[str, object],
                self._client.query(
                    TableName=self._config.sessions_table,
                    IndexName=_CALL_SID_INDEX,
                    KeyConditionExpression="call_sid = :call_sid",
                    ExpressionAttributeValues=self._serialize_item({":call_sid": call_sid}),
                ),
            )
        except ClientError as error:
            raise _persistence_error("SESSION_LOOKUP_FAILED", error) from error
        raw_items = cast(list[Mapping[str, object]], response.get("Items", []))
        if not raw_items:
            raise PersistenceNotFoundError("SESSION_NOT_FOUND")
        sessions = [self._deserialize_item(item) for item in raw_items]
        return max(sessions, key=lambda item: _required_str(item, "started_at"))

    def _conditional_put_persona(
        self,
        persona: Persona,
        *,
        expected_version: int,
    ) -> None:
        condition = (
            "attribute_not_exists(persona_id)" if expected_version == 0 else "#version = :expected"
        )
        kwargs: dict[str, object] = {
            "TableName": self._config.personas_table,
            "Item": self._serialize_item(_persona_to_item(persona)),
            "ConditionExpression": condition,
        }
        if expected_version > 0:
            kwargs["ExpressionAttributeNames"] = {"#version": "version"}
            kwargs["ExpressionAttributeValues"] = self._serialize_item(
                {":expected": expected_version}
            )
        try:
            self._client.put_item(**kwargs)
        except ClientError as error:
            if _error_code(error) == "ConditionalCheckFailedException":
                raise PersistenceConflictError("PERSONA_VERSION_CONFLICT") from error
            raise _persistence_error("PERSONA_WRITE_FAILED", error) from error

    def _replace_active_persona(
        self,
        persona: Persona,
        *,
        expected_version: int,
    ) -> None:
        transaction = [
            self._persona_put_transaction(persona, expected_version=expected_version),
            {
                "Update": {
                    "TableName": self._config.personas_table,
                    "Key": self._serialize_item({"persona_id": _ACTIVE_POINTER_ID}),
                    "UpdateExpression": "SET active_persona_version = :new",
                    "ConditionExpression": (
                        "active_persona_id = :id AND active_persona_version = :expected"
                    ),
                    "ExpressionAttributeValues": self._serialize_item(
                        {
                            ":new": persona.version,
                            ":id": persona.persona_id,
                            ":expected": expected_version,
                        }
                    ),
                }
            },
        ]
        try:
            self._client.transact_write_items(TransactItems=transaction)
        except ClientError as error:
            if _error_code(error) == "TransactionCanceledException":
                raise PersistenceConflictError("PERSONA_VERSION_CONFLICT") from error
            raise _persistence_error("PERSONA_WRITE_FAILED", error) from error

    def _persona_put_transaction(
        self,
        persona: Persona,
        *,
        expected_version: int,
    ) -> dict[str, object]:
        return {
            "Put": {
                "TableName": self._config.personas_table,
                "Item": self._serialize_item(_persona_to_item(persona)),
                "ConditionExpression": "#version = :expected",
                "ExpressionAttributeNames": {"#version": "version"},
                "ExpressionAttributeValues": self._serialize_item({":expected": expected_version}),
            }
        }

    def _pointer_put_transaction(
        self,
        persona: Persona,
        *,
        previous: Persona | None,
        pointer_exists: bool,
    ) -> dict[str, object]:
        put: dict[str, object] = {
            "TableName": self._config.personas_table,
            "Item": self._serialize_item(
                {
                    "persona_id": _ACTIVE_POINTER_ID,
                    "active_persona_id": persona.persona_id,
                    "active_persona_version": persona.version,
                    "updated_at": persona.updated_at.isoformat(),
                }
            ),
        }
        if not pointer_exists:
            put["ConditionExpression"] = "attribute_not_exists(persona_id)"
        else:
            if previous is None:
                raise PersistenceError("ACTIVE_PERSONA_POINTER_INVALID")
            put["ConditionExpression"] = (
                "active_persona_id = :previous_id AND active_persona_version = :previous_version"
            )
            put["ExpressionAttributeValues"] = self._serialize_item(
                {
                    ":previous_id": previous.persona_id,
                    ":previous_version": previous.version,
                }
            )
        return {"Put": put}

    def _get_item(
        self,
        table_name: str,
        key: dict[str, object],
        *,
        operation: str,
    ) -> dict[str, object] | None:
        try:
            response = cast(
                dict[str, object],
                self._client.get_item(
                    TableName=table_name,
                    Key=self._serialize_item(key),
                    ConsistentRead=True,
                ),
            )
        except ClientError as error:
            raise _persistence_error(operation, error) from error
        raw_item = response.get("Item")
        if not isinstance(raw_item, dict):
            return None
        return self._deserialize_item(cast(Mapping[str, object], raw_item))

    def _ensure_table(
        self,
        *,
        table_name: str,
        key_schema: list[dict[str, str]],
        attribute_definitions: list[dict[str, str]],
        global_secondary_indexes: list[dict[str, object]] | None = None,
    ) -> None:
        try:
            self._client.describe_table(TableName=table_name)
            return
        except ClientError as error:
            if _error_code(error) != "ResourceNotFoundException":
                raise _persistence_error("TABLE_DESCRIBE_FAILED", error) from error
        request: dict[str, object] = {
            "TableName": table_name,
            "BillingMode": "PAY_PER_REQUEST",
            "KeySchema": key_schema,
            "AttributeDefinitions": attribute_definitions,
        }
        if global_secondary_indexes is not None:
            request["GlobalSecondaryIndexes"] = global_secondary_indexes
        try:
            self._client.create_table(**request)
            self._client.get_waiter("table_exists").wait(TableName=table_name)
        except ClientError as error:
            raise _persistence_error("TABLE_CREATE_FAILED", error) from error

    def _ensure_ttl(self, table_name: str) -> None:
        try:
            response = cast(
                dict[str, object],
                self._client.describe_time_to_live(TableName=table_name),
            )
            description = response.get("TimeToLiveDescription")
            if isinstance(description, dict) and description.get("TimeToLiveStatus") in {
                "ENABLED",
                "ENABLING",
            }:
                return
            self._client.update_time_to_live(
                TableName=table_name,
                TimeToLiveSpecification={
                    "Enabled": True,
                    "AttributeName": _TTL_ATTRIBUTE,
                },
            )
        except ClientError as error:
            raise _persistence_error("TABLE_TTL_FAILED", error) from error

    def _serialize_item(self, item: Mapping[str, object]) -> dict[str, object]:
        return {key: self._serializer.serialize(value) for key, value in item.items()}

    def _deserialize_item(self, item: Mapping[str, object]) -> dict[str, object]:
        return {
            key: _normalize_dynamo_value(self._deserializer.deserialize(value))
            for key, value in item.items()
        }


def _persona_to_item(persona: Persona) -> dict[str, object]:
    return {
        "persona_id": persona.persona_id,
        "name": persona.name,
        "system_prompt": persona.system_prompt,
        "voice_id": persona.voice_id,
        "version": persona.version,
        "active": persona.active,
        "created_at": persona.created_at.isoformat(),
        "updated_at": persona.updated_at.isoformat(),
    }


def _persona_from_item(item: Mapping[str, object]) -> Persona:
    return Persona(
        persona_id=_required_str(item, "persona_id"),
        name=_required_str(item, "name"),
        system_prompt=_required_str(item, "system_prompt"),
        voice_id=_required_str(item, "voice_id"),
        version=_required_int(item, "version"),
        active=_required_bool(item, "active"),
        created_at=_required_datetime(item, "created_at"),
        updated_at=_required_datetime(item, "updated_at"),
    )


def _transcript_from_item(item: Mapping[str, object]) -> TranscriptTurn:
    try:
        speaker = TranscriptSpeaker(_required_str(item, "speaker"))
    except ValueError as error:
        raise PersistenceError("TRANSCRIPT_SPEAKER_INVALID") from error
    return TranscriptTurn(
        session_id=_required_str(item, "session_id"),
        turn_number=_required_int(item, "turn_number"),
        speaker=speaker,
        text=_required_str(item, "text"),
        source_event_id=_required_str(item, "source_event_id"),
        created_at=_required_datetime(item, "created_at"),
        expires_at=_required_int(item, "expires_at"),
    )


def _normalize_dynamo_value(value: object) -> object:
    if isinstance(value, Decimal):
        if value == value.to_integral_value():
            return int(value)
        return float(value)
    if isinstance(value, list):
        return [_normalize_dynamo_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _normalize_dynamo_value(item) for key, item in value.items()}
    return value


def _required_str(item: Mapping[str, object], field: str) -> str:
    value = item.get(field)
    if not isinstance(value, str) or not value:
        raise PersistenceError(f"PERSISTED_{field.upper()}_INVALID")
    return value


def _required_int(item: Mapping[str, object], field: str) -> int:
    value = item.get(field)
    if not isinstance(value, int) or isinstance(value, bool):
        raise PersistenceError(f"PERSISTED_{field.upper()}_INVALID")
    return value


def _required_bool(item: Mapping[str, object], field: str) -> bool:
    value = item.get(field)
    if not isinstance(value, bool):
        raise PersistenceError(f"PERSISTED_{field.upper()}_INVALID")
    return value


def _required_datetime(item: Mapping[str, object], field: str) -> datetime:
    value = _required_str(item, field)
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise PersistenceError(f"PERSISTED_{field.upper()}_INVALID") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise PersistenceError(f"PERSISTED_{field.upper()}_INVALID")
    return parsed


def _error_code(error: ClientError) -> str:
    return str(error.response.get("Error", {}).get("Code", "UNKNOWN"))


def _persistence_error(operation: str, error: ClientError) -> PersistenceError:
    code = _error_code(error)
    return PersistenceError(
        f"{operation}:{code}",
        code=code,
        retryable=code in _RETRYABLE_REJECTION_CODES,
    )
