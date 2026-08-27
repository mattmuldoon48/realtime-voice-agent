"""Bounded observability values shared by logging, metrics, and workers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final, Literal, Protocol


class MetricName(StrEnum):
    """Fixed custom metric names for the voice-agent service."""

    CALLS_STARTED = "CallsStarted"
    CALLS_COMPLETED = "CallsCompleted"
    CALLS_FAILED = "CallsFailed"
    CALL_START_TO_FIRST_AUDIO_MS = "CallStartToFirstAudioMs"
    CALL_DURATION_MS = "CallDurationMs"
    NOVA_ERRORS = "NovaErrors"
    TWILIO_ERRORS = "TwilioErrors"
    PERSISTENCE_ERRORS = "PersistenceErrors"
    PERSISTENCE_RETRIES = "PersistenceRetries"
    TRANSCRIPT_TURNS_PERSISTED = "TranscriptTurnsPersisted"
    BARGE_INS = "BargeIns"
    CONTINUATION_ATTEMPTS = "ContinuationAttempts"
    CONTINUATION_SUCCESSES = "ContinuationSuccesses"
    CONTINUATION_FAILURES = "ContinuationFailures"


class MetricComponent(StrEnum):
    """Bounded component dimension values."""

    CALL = "Call"
    NOVA = "Nova"
    TWILIO = "Twilio"
    PERSISTENCE = "Persistence"
    TRANSCRIPT = "Transcript"


class MetricOutcome(StrEnum):
    """Bounded terminal outcome dimension values."""

    SUCCEEDED = "SUCCEEDED"
    DISCONNECTED = "DISCONNECTED"
    FAILED = "FAILED"


MetricUnit = Literal["Count", "Milliseconds"]

_ALLOWED_DIMENSION_NAMES: Final = frozenset({"Environment", "Component", "Outcome"})
_ALLOWED_COMPONENT_VALUES: Final = frozenset(component.value for component in MetricComponent)
_ALLOWED_OUTCOME_VALUES: Final = frozenset(outcome.value for outcome in MetricOutcome)


@dataclass(frozen=True, slots=True)
class MetricDimension:
    """One validated low-cardinality CloudWatch metric dimension."""

    name: str
    value: str

    def __post_init__(self) -> None:
        if self.name not in _ALLOWED_DIMENSION_NAMES:
            raise ValueError(f"Unsupported metric dimension: {self.name}")
        if not self.value or len(self.value) > 64:
            raise ValueError("Metric dimension values must contain 1 to 64 characters")
        if self.name == "Component" and self.value not in _ALLOWED_COMPONENT_VALUES:
            raise ValueError(f"Unsupported metric component: {self.value}")
        if self.name == "Outcome" and self.value not in _ALLOWED_OUTCOME_VALUES:
            raise ValueError(f"Unsupported metric outcome: {self.value}")


@dataclass(frozen=True, slots=True)
class MetricDatum:
    """One validated custom metric queued for asynchronous delivery."""

    name: MetricName
    value: float
    unit: MetricUnit
    dimensions: tuple[MetricDimension, ...]

    def __post_init__(self) -> None:
        names = tuple(dimension.name for dimension in self.dimensions)
        if len(names) != len(set(names)):
            raise ValueError("Metric dimensions must be unique")
        if "Environment" not in names or "Component" not in names:
            raise ValueError("Metrics require Environment and Component dimensions")


class TelemetryPublisher(Protocol):
    """Non-blocking publishing boundary safe to call from media loops."""

    def publish_metric(self, metric: MetricDatum) -> bool:
        """Queue one metric without waiting for network I/O."""

    def publish_log(self, event: dict[str, object]) -> bool:
        """Queue one already-sanitized structured log without network I/O."""


class NullTelemetryPublisher:
    """Disabled publisher that performs no work and never fails callers."""

    def publish_metric(self, metric: MetricDatum) -> bool:
        del metric
        return True

    def publish_log(self, event: dict[str, object]) -> bool:
        del event
        return True

    async def start(self) -> None:
        """Start no worker because telemetry is disabled."""

    async def close(self) -> None:
        """Flush no data because telemetry is disabled."""
