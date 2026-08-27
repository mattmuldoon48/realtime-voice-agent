"""Exactly-once per-call metric emission."""

from __future__ import annotations

from realtime_voice_agent.observability.models import (
    MetricComponent,
    MetricDatum,
    MetricDimension,
    MetricName,
    MetricOutcome,
    TelemetryPublisher,
)


class CallMetrics:
    """Own metric deduplication state for one call session."""

    def __init__(self, *, publisher: TelemetryPublisher, environment: str) -> None:
        self._publisher = publisher
        self._environment = environment
        self._started = False
        self._call_start_to_first_audio = False
        self._terminal = False
        self._errors: set[MetricComponent] = set()

    def call_started(self) -> None:
        """Attempt the CallsStarted metric once."""
        if self._started:
            return
        self._started = True
        self._count(MetricName.CALLS_STARTED, MetricComponent.CALL)

    def call_start_to_first_audio(self, *, elapsed_ms: int) -> None:
        """Publish call start to first Nova audio once."""
        if self._call_start_to_first_audio:
            return
        self._call_start_to_first_audio = True
        self._publish(
            MetricDatum(
                name=MetricName.CALL_START_TO_FIRST_AUDIO_MS,
                value=float(max(elapsed_ms, 0)),
                unit="Milliseconds",
                dimensions=self._dimensions(MetricComponent.CALL),
            )
        )

    def transcript_turn_persisted(self) -> None:
        """Count one final transcript turn after its successful write."""
        self._count(MetricName.TRANSCRIPT_TURNS_PERSISTED, MetricComponent.TRANSCRIPT)

    def persistence_retry(self) -> None:
        """Count one scheduled retry without adding per-call dimensions."""
        self._count(MetricName.PERSISTENCE_RETRIES, MetricComponent.PERSISTENCE)

    def barge_in(self) -> None:
        """Count one caller interruption without adding per-call dimensions."""
        self._count(MetricName.BARGE_INS, MetricComponent.CALL)

    def continuation_attempted(self) -> None:
        """Count one bounded Nova connection replacement attempt."""
        self._count(MetricName.CONTINUATION_ATTEMPTS, MetricComponent.NOVA)

    def continuation_succeeded(self) -> None:
        """Count one successful Nova connection handoff."""
        self._count(MetricName.CONTINUATION_SUCCESSES, MetricComponent.NOVA)

    def continuation_failed(self) -> None:
        """Count one failed replacement or retired-stream cleanup."""
        self._count(MetricName.CONTINUATION_FAILURES, MetricComponent.NOVA)

    def component_error(self, component: MetricComponent) -> None:
        """Attempt one error counter per component for this call."""
        if component in self._errors:
            return
        error_name = {
            MetricComponent.NOVA: MetricName.NOVA_ERRORS,
            MetricComponent.TWILIO: MetricName.TWILIO_ERRORS,
            MetricComponent.PERSISTENCE: MetricName.PERSISTENCE_ERRORS,
        }.get(component)
        if error_name is None:
            raise ValueError(f"{component} does not have an error metric")
        self._errors.add(component)
        self._count(error_name, component)

    def call_terminal(
        self,
        *,
        outcome: MetricOutcome,
        duration_ms: int,
        failure_code: str | None,
    ) -> None:
        """Attempt terminal counters and duration exactly once."""
        if self._terminal:
            return
        self._terminal = True
        dimensions = self._dimensions(MetricComponent.CALL, outcome=outcome)
        terminal_name = (
            MetricName.CALLS_FAILED
            if outcome is MetricOutcome.FAILED
            else MetricName.CALLS_COMPLETED
        )
        self._publish(
            MetricDatum(
                name=terminal_name,
                value=1.0,
                unit="Count",
                dimensions=dimensions,
            )
        )
        self._publish(
            MetricDatum(
                name=MetricName.CALL_DURATION_MS,
                value=float(max(duration_ms, 0)),
                unit="Milliseconds",
                dimensions=dimensions,
            )
        )
        if outcome is MetricOutcome.FAILED and failure_code is not None:
            component = _failure_component(failure_code)
            if component is not None:
                self.component_error(component)

    def _count(self, name: MetricName, component: MetricComponent) -> None:
        self._publish(
            MetricDatum(
                name=name,
                value=1.0,
                unit="Count",
                dimensions=self._dimensions(component),
            )
        )

    def _dimensions(
        self,
        component: MetricComponent,
        *,
        outcome: MetricOutcome | None = None,
    ) -> tuple[MetricDimension, ...]:
        dimensions = (
            MetricDimension(name="Environment", value=self._environment),
            MetricDimension(name="Component", value=component.value),
        )
        if outcome is None:
            return dimensions
        return (*dimensions, MetricDimension(name="Outcome", value=outcome.value))

    def _publish(self, metric: MetricDatum) -> None:
        self._publisher.publish_metric(metric)


def _failure_component(failure_code: str) -> MetricComponent | None:
    if failure_code.startswith("NOVA_"):
        return MetricComponent.NOVA
    if failure_code.startswith("PERSISTENCE_"):
        return MetricComponent.PERSISTENCE
    if failure_code.startswith("TWILIO_"):
        return MetricComponent.TWILIO
    return None
