"""Unit tests for bounded structured logging and CloudWatch telemetry."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import cast

import pytest

from realtime_voice_agent.config import ObservabilityRuntimeConfig
from realtime_voice_agent.observability.bootstrap import (
    bootstrap_observability,
    build_reliability_dashboard,
    dashboard_name,
)
from realtime_voice_agent.observability.call_metrics import CallMetrics
from realtime_voice_agent.observability.cloudwatch import CloudWatchTelemetry
from realtime_voice_agent.observability.logging import (
    configure_local_logging,
    get_logger,
    sanitize_event,
)
from realtime_voice_agent.observability.models import (
    MetricComponent,
    MetricDatum,
    MetricDimension,
    MetricName,
    MetricOutcome,
)


class FakePublisher:
    def __init__(self) -> None:
        self.metrics: list[MetricDatum] = []
        self.logs: list[dict[str, object]] = []

    def publish_metric(self, metric: MetricDatum) -> bool:
        self.metrics.append(metric)
        return True

    def publish_log(self, event: dict[str, object]) -> bool:
        self.logs.append(event)
        return True


class FakeLogsClient:
    def __init__(self, *, fail_puts: bool = False) -> None:
        self.fail_puts = fail_puts
        self.stream_requests: list[dict[str, object]] = []
        self.put_requests: list[dict[str, object]] = []

    def create_log_stream(self, **kwargs: object) -> Mapping[str, object]:
        self.stream_requests.append(kwargs)
        return {}

    def put_log_events(self, **kwargs: object) -> Mapping[str, object]:
        self.put_requests.append(kwargs)
        if self.fail_puts:
            raise RuntimeError("synthetic logs failure")
        return {}


class FakeMetricsClient:
    def __init__(self, *, fail_puts: bool = False) -> None:
        self.fail_puts = fail_puts
        self.put_requests: list[dict[str, object]] = []

    def put_metric_data(self, **kwargs: object) -> Mapping[str, object]:
        self.put_requests.append(kwargs)
        if self.fail_puts:
            raise RuntimeError("synthetic metrics failure")
        return {}


def _config(
    *,
    queue_max_events: int = 10,
    batch_max_items: int = 10,
    max_attempts: int = 1,
) -> ObservabilityRuntimeConfig:
    return ObservabilityRuntimeConfig(
        enabled=True,
        region="us-east-1",
        profile="test-profile",
        environment="test",
        service_name="realtime-voice-agent",
        log_group="/realtime-voice-agent/test",
        log_stream="local-test",
        metric_namespace="RealtimeVoiceAgent/VoiceAgent",
        queue_max_events=queue_max_events,
        batch_max_items=batch_max_items,
        flush_interval_seconds=0.01,
        max_attempts=max_attempts,
        cleanup_timeout_seconds=1.0,
    )


def _metric(name: MetricName = MetricName.CALLS_STARTED) -> MetricDatum:
    return MetricDatum(
        name=name,
        value=1.0,
        unit="Count",
        dimensions=(
            MetricDimension(name="Environment", value="test"),
            MetricDimension(name="Component", value="Call"),
        ),
    )


def test_structured_logging_emits_same_sanitized_json_locally_and_to_sink(
    capsys: pytest.CaptureFixture[str],
) -> None:
    publisher = FakePublisher()
    configure_local_logging(
        level="INFO",
        service="realtime-voice-agent",
        cloud_log_sink=publisher.publish_log,
    )

    get_logger().info(
        "call_session_terminal",
        session_id="session-1",
        outcome="SUCCEEDED",
        auth_token="not-a-real-secret",
    )

    expected_redaction = "[REDACTED]"
    document = json.loads(capsys.readouterr().out)
    assert document["service"] == "realtime-voice-agent"
    assert document["event"] == "call_session_terminal"
    assert document["level"] == "info"
    assert document["timestamp"].endswith("Z")
    assert document["session_id"] == "session-1"
    assert document["outcome"] == "SUCCEEDED"
    assert document["auth_token"] == expected_redaction
    assert publisher.logs == [document]


def test_sensitive_fields_are_recursively_redacted_without_hiding_counts() -> None:
    event = sanitize_event(
        {
            "phone_number": "+15555550100",
            "system_prompt": "private instructions",
            "transcript_text": "private transcript",
            "headers": {"Authorization": "Bearer value"},
            "nested": {"credential": "value"},
            "pcm16_bytes": 320,
            "outbound_payload_base64_bytes": 216,
        }
    )

    assert event["phone_number"] == "[REDACTED]"
    assert event["system_prompt"] == "[REDACTED]"
    assert event["transcript_text"] == "[REDACTED]"
    assert event["headers"] == "[REDACTED]"
    assert event["nested"] == {"credential": "[REDACTED]"}
    assert event["pcm16_bytes"] == 320
    assert event["outbound_payload_base64_bytes"] == 216


def test_metric_dimensions_reject_high_cardinality_identifiers() -> None:
    with pytest.raises(ValueError, match="Unsupported metric dimension"):
        MetricDimension(name="SessionId", value="session-1")

    with pytest.raises(ValueError, match="Unsupported metric component"):
        MetricDimension(name="Component", value="session-1")


def test_call_metrics_emit_terminal_and_component_errors_exactly_once() -> None:
    publisher = FakePublisher()
    metrics = CallMetrics(publisher=publisher, environment="test")

    metrics.call_started()
    metrics.call_started()
    metrics.call_start_to_first_audio(elapsed_ms=125)
    metrics.call_start_to_first_audio(elapsed_ms=999)
    metrics.barge_in()
    metrics.barge_in()
    metrics.continuation_attempted()
    metrics.continuation_succeeded()
    metrics.continuation_failed()
    metrics.persistence_retry()
    metrics.persistence_retry()
    metrics.call_terminal(
        outcome=MetricOutcome.FAILED,
        duration_ms=500,
        failure_code="NOVA_STREAM_FAILED",
    )
    metrics.call_terminal(
        outcome=MetricOutcome.FAILED,
        duration_ms=900,
        failure_code="NOVA_STREAM_FAILED",
    )
    metrics.component_error(MetricComponent.NOVA)

    names = [metric.name for metric in publisher.metrics]
    assert names.count(MetricName.CALLS_STARTED) == 1
    assert names.count(MetricName.CALL_START_TO_FIRST_AUDIO_MS) == 1
    assert names.count(MetricName.CALLS_FAILED) == 1
    assert names.count(MetricName.CALL_DURATION_MS) == 1
    assert names.count(MetricName.NOVA_ERRORS) == 1
    assert names.count(MetricName.BARGE_INS) == 2
    assert names.count(MetricName.CONTINUATION_ATTEMPTS) == 1
    assert names.count(MetricName.CONTINUATION_SUCCESSES) == 1
    assert names.count(MetricName.CONTINUATION_FAILURES) == 1
    assert names.count(MetricName.PERSISTENCE_RETRIES) == 2
    assert all(
        {dimension.name for dimension in metric.dimensions}
        <= {"Environment", "Component", "Outcome"}
        for metric in publisher.metrics
    )


def test_telemetry_queue_overflow_is_nonfatal_and_reported_once(
    capsys: pytest.CaptureFixture[str],
) -> None:
    telemetry = CloudWatchTelemetry(
        config=_config(queue_max_events=1),
        logs_client=FakeLogsClient(),
        metrics_client=FakeMetricsClient(),
        clock_ms=lambda: 1,
    )

    assert telemetry.publish_metric(_metric()) is True
    assert telemetry.publish_metric(_metric()) is False
    assert telemetry.publish_metric(_metric()) is False

    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [event["event"] for event in events] == ["telemetry_queue_overflow"]


@pytest.mark.asyncio
async def test_worker_batches_logs_and_metrics_and_flushes_on_shutdown() -> None:
    logs = FakeLogsClient()
    metrics = FakeMetricsClient()
    telemetry = CloudWatchTelemetry(
        config=_config(),
        logs_client=logs,
        metrics_client=metrics,
        clock_ms=lambda: 1_000,
    )
    await telemetry.start()

    assert telemetry.publish_log({"event": "one"}) is True
    assert telemetry.publish_log({"event": "two"}) is True
    assert telemetry.publish_metric(_metric()) is True
    assert telemetry.publish_metric(_metric(MetricName.TRANSCRIPT_TURNS_PERSISTED)) is True
    await telemetry.close()

    assert len(logs.stream_requests) == 1
    assert len(logs.put_requests) == 1
    assert len(cast(list[object], logs.put_requests[0]["logEvents"])) == 2
    assert len(metrics.put_requests) == 1
    assert len(cast(list[object], metrics.put_requests[0]["MetricData"])) == 2
    assert telemetry.publish_log({"event": "after-close"}) is False


@pytest.mark.asyncio
async def test_cloudwatch_client_failures_retry_and_do_not_escape_worker(
    capsys: pytest.CaptureFixture[str],
) -> None:
    logs = FakeLogsClient(fail_puts=True)
    metrics = FakeMetricsClient(fail_puts=True)
    telemetry = CloudWatchTelemetry(
        config=_config(max_attempts=2),
        logs_client=logs,
        metrics_client=metrics,
        clock_ms=lambda: 1_000,
    )
    await telemetry.start()
    telemetry.publish_log({"event": "safe"})
    telemetry.publish_metric(_metric())

    await telemetry.close()

    assert len(logs.put_requests) == 2
    assert len(metrics.put_requests) == 2
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [event["event"] for event in events] == [
        "cloudwatch_delivery_failed",
        "cloudwatch_delivery_failed",
        "cloudwatch_delivery_failed",
        "cloudwatch_delivery_failed",
    ]
    assert all("synthetic" not in json.dumps(event) for event in events)


class AlreadyExistsError(Exception):
    def __init__(self) -> None:
        self.response = {"Error": {"Code": "ResourceAlreadyExistsException"}}


class FakeBootstrapLogs:
    def __init__(self) -> None:
        self.groups: set[str] = set()
        self.streams: set[tuple[str, str]] = set()
        self.retention_requests: list[dict[str, object]] = []

    def create_log_group(self, **kwargs: object) -> Mapping[str, object]:
        group = cast(str, kwargs["logGroupName"])
        if group in self.groups:
            raise AlreadyExistsError
        self.groups.add(group)
        return {}

    def put_retention_policy(self, **kwargs: object) -> Mapping[str, object]:
        self.retention_requests.append(kwargs)
        return {}

    def create_log_stream(self, **kwargs: object) -> Mapping[str, object]:
        key = (cast(str, kwargs["logGroupName"]), cast(str, kwargs["logStreamName"]))
        if key in self.streams:
            raise AlreadyExistsError
        self.streams.add(key)
        return {}


class FakeBootstrapMetrics:
    def __init__(self) -> None:
        self.alarm_requests: list[dict[str, object]] = []
        self.dashboard_requests: list[dict[str, object]] = []
        self.dashboard_body: str | None = None

    def put_metric_alarm(self, **kwargs: object) -> Mapping[str, object]:
        self.alarm_requests.append(kwargs)
        return {}

    def describe_alarms(self, **kwargs: object) -> Mapping[str, object]:
        names = cast(list[str], kwargs["AlarmNames"])
        return {"MetricAlarms": [{"AlarmName": name} for name in names]}

    def put_dashboard(self, **kwargs: object) -> Mapping[str, object]:
        self.dashboard_requests.append(kwargs)
        self.dashboard_body = cast(str, kwargs["DashboardBody"])
        return {"DashboardValidationMessages": []}

    def get_dashboard(self, **kwargs: object) -> Mapping[str, object]:
        del kwargs
        if self.dashboard_body is None:
            raise RuntimeError("dashboard was not created")
        return {"DashboardBody": self.dashboard_body}


def test_observability_bootstrap_is_safe_to_rerun_with_identical_configuration() -> None:
    logs = FakeBootstrapLogs()
    metrics = FakeBootstrapMetrics()

    first = bootstrap_observability(
        config=_config(),
        logs_client=logs,
        metrics_client=metrics,
    )
    second = bootstrap_observability(
        config=_config(),
        logs_client=logs,
        metrics_client=metrics,
    )

    assert first == second
    assert first.retention_days == 7
    assert first.inspected_alarm_count == 2
    assert len(logs.groups) == 1
    assert len(logs.streams) == 1
    assert len(logs.retention_requests) == 2
    assert metrics.alarm_requests[:2] == metrics.alarm_requests[2:]
    assert metrics.alarm_requests[0]["Threshold"] == 1.0
    assert metrics.alarm_requests[1]["Threshold"] == 3_000.0
    assert metrics.alarm_requests[1]["AlarmName"].endswith("-call-start-to-first-audio")
    assert metrics.alarm_requests[1]["MetricName"] == "CallStartToFirstAudioMs"
    assert metrics.alarm_requests[1]["Dimensions"] == [
        {"Name": "Environment", "Value": "test"},
        {"Name": "Component", "Value": "Call"},
    ]
    assert first.first_audio_alarm.endswith("-call-start-to-first-audio")
    assert first.dashboard_name == "realtime-voice-agent-test-reliability"
    assert first.dashboard_widget_count == 4
    assert metrics.dashboard_requests[0] == metrics.dashboard_requests[1]
    dashboard = build_reliability_dashboard(_config())
    assert dashboard_name(_config()) == first.dashboard_name
    assert json.loads(cast(str, metrics.dashboard_requests[0]["DashboardBody"])) == dashboard
    dashboard_json = json.dumps(dashboard)
    assert "PersistenceRetries" in dashboard_json
    assert "CallStartToFirstAudioMs" in dashboard_json
    assert "FirstResponseLatencyMs" not in dashboard_json
    assert "ContinuationFailures" in dashboard_json
    assert "session_id" not in dashboard_json
    assert "call_sid" not in dashboard_json
    assert "transcript" not in dashboard_json.lower()
