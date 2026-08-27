"""Bounded read-only AWS dependency readiness checks."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol, cast

import boto3  # type: ignore[import-untyped]
from botocore.config import Config as BotoClientConfig  # type: ignore[import-untyped]

from realtime_voice_agent.config import PersistenceRuntimeConfig, ReadinessRuntimeConfig
from realtime_voice_agent.persistence.dynamodb import DynamoPersistenceStore
from realtime_voice_agent.persistence.errors import PersistenceNotFoundError
from realtime_voice_agent.persistence.ports import PersonaRepository

_CALL_SID_INDEX = "CallSidIndex"
_TTL_ATTRIBUTE = "expires_at"


class ReadinessErrorCode(StrEnum):
    """Safe bounded readiness failures suitable for HTTP and CLI output."""

    AWS_IDENTITY_UNAVAILABLE = "AWS_IDENTITY_UNAVAILABLE"
    ROOT_AWS_IDENTITY = "ROOT_AWS_IDENTITY"
    TABLE_MISSING = "TABLE_MISSING"
    TABLE_UNAVAILABLE = "TABLE_UNAVAILABLE"
    TABLE_INACTIVE = "TABLE_INACTIVE"
    TABLE_KEY_SCHEMA_INVALID = "TABLE_KEY_SCHEMA_INVALID"
    CALL_SID_INDEX_MISSING = "CALL_SID_INDEX_MISSING"
    CALL_SID_INDEX_INACTIVE = "CALL_SID_INDEX_INACTIVE"
    CALL_SID_INDEX_INVALID = "CALL_SID_INDEX_INVALID"
    TTL_CONFIGURATION_INVALID = "TTL_CONFIGURATION_INVALID"
    ACTIVE_PERSONA_MISSING = "ACTIVE_PERSONA_MISSING"
    ACTIVE_PERSONA_UNAVAILABLE = "ACTIVE_PERSONA_UNAVAILABLE"
    READINESS_TIMEOUT = "READINESS_TIMEOUT"
    READINESS_CHECK_FAILED = "READINESS_CHECK_FAILED"


@dataclass(frozen=True, slots=True)
class ReadinessResult:
    """Safe readiness result containing no AWS or application identifiers."""

    is_ready: bool
    error_code: ReadinessErrorCode | None = None

    @classmethod
    def ready(cls) -> ReadinessResult:
        """Return a successful readiness result."""
        return cls(is_ready=True)

    @classmethod
    def unavailable(cls, error_code: ReadinessErrorCode) -> ReadinessResult:
        """Return one bounded dependency failure."""
        return cls(is_ready=False, error_code=error_code)

    def to_document(self) -> dict[str, str]:
        """Serialize only the safe readiness status and bounded error code."""
        if self.is_ready:
            return {"status": "ready"}
        if self.error_code is None:
            return {
                "status": "not_ready",
                "error_code": ReadinessErrorCode.READINESS_CHECK_FAILED.value,
            }
        return {"status": "not_ready", "error_code": self.error_code.value}


class ReadinessChecker(Protocol):
    """Shared asynchronous readiness boundary used by HTTP and CLI entrypoints."""

    async def check(self) -> ReadinessResult: ...


class AwsIdentityClient(Protocol):
    """Read-only STS operation required by readiness."""

    def get_caller_identity(self, **kwargs: object) -> Mapping[str, object]: ...


class DynamoReadinessClient(Protocol):
    """Read-only DynamoDB metadata operations required by readiness."""

    def describe_table(self, **kwargs: object) -> Mapping[str, object]: ...

    def describe_time_to_live(self, **kwargs: object) -> Mapping[str, object]: ...


@dataclass(frozen=True, slots=True)
class _TableContract:
    name: str
    key_schema: frozenset[tuple[str, str]]
    attribute_types: Mapping[str, str]


class DependencyReadinessService:
    """Validate identity, DynamoDB metadata, TTL, and the active persona."""

    def __init__(
        self,
        *,
        config: ReadinessRuntimeConfig,
        identity_client: AwsIdentityClient,
        dynamodb_client: DynamoReadinessClient,
        persona_repository: PersonaRepository,
    ) -> None:
        self._config = config
        self._identity_client = identity_client
        self._dynamodb_client = dynamodb_client
        self._persona_repository = persona_repository

    async def check(self) -> ReadinessResult:
        """Run the complete read-only dependency check within one overall timeout."""
        try:
            async with asyncio.timeout(self._config.timeout_seconds):
                return await asyncio.to_thread(self._check_sync)
        except TimeoutError:
            return ReadinessResult.unavailable(ReadinessErrorCode.READINESS_TIMEOUT)
        except Exception:
            return ReadinessResult.unavailable(ReadinessErrorCode.READINESS_CHECK_FAILED)

    def _check_sync(self) -> ReadinessResult:
        identity_failure = self._check_identity()
        if identity_failure is not None:
            return identity_failure

        contracts = (
            _TableContract(
                name=self._config.personas_table,
                key_schema=frozenset({("persona_id", "HASH")}),
                attribute_types={"persona_id": "S"},
            ),
            _TableContract(
                name=self._config.sessions_table,
                key_schema=frozenset({("session_id", "HASH")}),
                attribute_types={"session_id": "S", "call_sid": "S"},
            ),
            _TableContract(
                name=self._config.transcripts_table,
                key_schema=frozenset(
                    {
                        ("session_id", "HASH"),
                        ("turn_number", "RANGE"),
                    }
                ),
                attribute_types={"session_id": "S", "turn_number": "N"},
            ),
        )
        table_documents: dict[str, Mapping[str, object]] = {}
        for contract in contracts:
            result, table = self._check_table(contract)
            if result is not None:
                return result
            if table is None:
                return ReadinessResult.unavailable(ReadinessErrorCode.TABLE_UNAVAILABLE)
            table_documents[contract.name] = table

        sessions = table_documents[self._config.sessions_table]
        index_failure = _check_call_sid_index(sessions)
        if index_failure is not None:
            return index_failure

        for table_name in (self._config.sessions_table, self._config.transcripts_table):
            ttl_failure = self._check_ttl(table_name)
            if ttl_failure is not None:
                return ttl_failure

        try:
            persona = self._persona_repository.get_active_persona()
            if not persona.active:
                return ReadinessResult.unavailable(ReadinessErrorCode.ACTIVE_PERSONA_UNAVAILABLE)
            persona.snapshot()
        except PersistenceNotFoundError:
            return ReadinessResult.unavailable(ReadinessErrorCode.ACTIVE_PERSONA_MISSING)
        except Exception:
            return ReadinessResult.unavailable(ReadinessErrorCode.ACTIVE_PERSONA_UNAVAILABLE)
        return ReadinessResult.ready()

    def _check_identity(self) -> ReadinessResult | None:
        try:
            response = self._identity_client.get_caller_identity()
        except Exception:
            return ReadinessResult.unavailable(ReadinessErrorCode.AWS_IDENTITY_UNAVAILABLE)
        arn = response.get("Arn")
        if not isinstance(arn, str) or not arn:
            return ReadinessResult.unavailable(ReadinessErrorCode.AWS_IDENTITY_UNAVAILABLE)
        if arn.endswith(":root"):
            return ReadinessResult.unavailable(ReadinessErrorCode.ROOT_AWS_IDENTITY)
        return None

    def _check_table(
        self,
        contract: _TableContract,
    ) -> tuple[ReadinessResult | None, Mapping[str, object] | None]:
        try:
            response = self._dynamodb_client.describe_table(TableName=contract.name)
        except Exception as error:
            code = _aws_error_code(error)
            if code == "ResourceNotFoundException":
                return ReadinessResult.unavailable(ReadinessErrorCode.TABLE_MISSING), None
            return ReadinessResult.unavailable(ReadinessErrorCode.TABLE_UNAVAILABLE), None
        table = response.get("Table")
        if not isinstance(table, Mapping):
            return ReadinessResult.unavailable(ReadinessErrorCode.TABLE_UNAVAILABLE), None
        if table.get("TableStatus") != "ACTIVE":
            return ReadinessResult.unavailable(ReadinessErrorCode.TABLE_INACTIVE), None
        if _key_schema(table.get("KeySchema")) != contract.key_schema:
            return ReadinessResult.unavailable(ReadinessErrorCode.TABLE_KEY_SCHEMA_INVALID), None
        attribute_types = _attribute_types(table.get("AttributeDefinitions"))
        if any(
            attribute_types.get(name) != value for name, value in contract.attribute_types.items()
        ):
            return ReadinessResult.unavailable(ReadinessErrorCode.TABLE_KEY_SCHEMA_INVALID), None
        return None, table

    def _check_ttl(self, table_name: str) -> ReadinessResult | None:
        try:
            response = self._dynamodb_client.describe_time_to_live(TableName=table_name)
        except Exception:
            return ReadinessResult.unavailable(ReadinessErrorCode.TTL_CONFIGURATION_INVALID)
        description = response.get("TimeToLiveDescription")
        if not isinstance(description, Mapping):
            return ReadinessResult.unavailable(ReadinessErrorCode.TTL_CONFIGURATION_INVALID)
        if (
            description.get("TimeToLiveStatus") != "ENABLED"
            or description.get("AttributeName") != _TTL_ATTRIBUTE
        ):
            return ReadinessResult.unavailable(ReadinessErrorCode.TTL_CONFIGURATION_INVALID)
        return None


class _StaticReadinessService:
    def __init__(self, result: ReadinessResult) -> None:
        self._result = result

    async def check(self) -> ReadinessResult:
        return self._result


def create_readiness_service(
    config: ReadinessRuntimeConfig,
    persistence_config: PersistenceRuntimeConfig,
) -> ReadinessChecker:
    """Create bounded standard-chain clients and the shared readiness service."""
    try:
        session = boto3.Session(profile_name=config.profile, region_name=config.region)
        operation_timeout = min(config.timeout_seconds, 2.0)
        client_config = BotoClientConfig(
            connect_timeout=operation_timeout,
            read_timeout=operation_timeout,
            retries={"total_max_attempts": 1, "mode": "standard"},
        )
        identity_client = cast(
            AwsIdentityClient,
            cast(Any, session.client("sts", config=client_config)),
        )
        dynamodb_client = cast(
            DynamoReadinessClient,
            cast(Any, session.client("dynamodb", config=client_config)),
        )
        persona_repository = DynamoPersistenceStore(
            persistence_config,
            client=dynamodb_client,
        )
    except Exception:
        return _StaticReadinessService(
            ReadinessResult.unavailable(ReadinessErrorCode.READINESS_CHECK_FAILED)
        )
    return DependencyReadinessService(
        config=config,
        identity_client=identity_client,
        dynamodb_client=dynamodb_client,
        persona_repository=persona_repository,
    )


def _check_call_sid_index(table: Mapping[str, object]) -> ReadinessResult | None:
    indexes = table.get("GlobalSecondaryIndexes")
    if not isinstance(indexes, list):
        return ReadinessResult.unavailable(ReadinessErrorCode.CALL_SID_INDEX_MISSING)
    match: Mapping[str, object] | None = None
    for value in indexes:
        if isinstance(value, Mapping) and value.get("IndexName") == _CALL_SID_INDEX:
            match = value
            break
    if match is None:
        return ReadinessResult.unavailable(ReadinessErrorCode.CALL_SID_INDEX_MISSING)
    if match.get("IndexStatus") != "ACTIVE":
        return ReadinessResult.unavailable(ReadinessErrorCode.CALL_SID_INDEX_INACTIVE)
    if _key_schema(match.get("KeySchema")) != frozenset({("call_sid", "HASH")}):
        return ReadinessResult.unavailable(ReadinessErrorCode.CALL_SID_INDEX_INVALID)
    return None


def _key_schema(value: object) -> frozenset[tuple[str, str]] | None:
    if not isinstance(value, list):
        return None
    schema: set[tuple[str, str]] = set()
    for item in value:
        if not isinstance(item, Mapping):
            return None
        name = item.get("AttributeName")
        key_type = item.get("KeyType")
        if not isinstance(name, str) or not isinstance(key_type, str):
            return None
        schema.add((name, key_type))
    return frozenset(schema)


def _attribute_types(value: object) -> dict[str, str]:
    if not isinstance(value, list):
        return {}
    attributes: dict[str, str] = {}
    for item in value:
        if not isinstance(item, Mapping):
            return {}
        name = item.get("AttributeName")
        attribute_type = item.get("AttributeType")
        if not isinstance(name, str) or not isinstance(attribute_type, str):
            return {}
        attributes[name] = attribute_type
    return attributes


def _aws_error_code(error: Exception) -> str | None:
    response = getattr(error, "response", None)
    if not isinstance(response, Mapping):
        return None
    details = response.get("Error")
    if not isinstance(details, Mapping):
        return None
    code = details.get("Code")
    return code if isinstance(code, str) else None
