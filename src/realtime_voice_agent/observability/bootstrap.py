"""Idempotent CloudWatch log and alarm bootstrap operations."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Protocol, cast

import boto3  # type: ignore[import-untyped]
from pydantic import ValidationError

from realtime_voice_agent.config import AppSettings, ConfigurationError, ObservabilityRuntimeConfig


class BootstrapLogsClient(Protocol):
    """CloudWatch Logs operations used by bootstrap."""

    def create_log_group(self, **kwargs: object) -> Mapping[str, object]: ...

    def put_retention_policy(self, **kwargs: object) -> Mapping[str, object]: ...

    def create_log_stream(self, **kwargs: object) -> Mapping[str, object]: ...


class BootstrapMetricsClient(Protocol):
    """CloudWatch alarm operations used by bootstrap."""

    def put_metric_alarm(self, **kwargs: object) -> Mapping[str, object]: ...

    def describe_alarms(self, **kwargs: object) -> Mapping[str, object]: ...
    def put_dashboard(self, **kwargs: object) -> Mapping[str, object]: ...

    def get_dashboard(self, **kwargs: object) -> Mapping[str, object]: ...


@dataclass(frozen=True, slots=True)
class BootstrapResult:
    """Safe summary of reproducibly configured CloudWatch resources."""

    log_group: str
    log_stream: str
    retention_days: int
    metric_namespace: str
    failure_alarm: str
    first_audio_alarm: str
    inspected_alarm_count: int
    dashboard_name: str
    dashboard_widget_count: int


def bootstrap_observability(
    *,
    config: ObservabilityRuntimeConfig,
    logs_client: BootstrapLogsClient,
    metrics_client: BootstrapMetricsClient,
) -> BootstrapResult:
    """Create or update the required log resources, alarms, and dashboard."""
    _create_log_group_if_needed(logs_client, config.log_group)
    logs_client.put_retention_policy(
        logGroupName=config.log_group,
        retentionInDays=7,
    )
    _create_log_stream_if_needed(logs_client, config.log_group, config.log_stream)

    failure_alarm, first_audio_alarm = alarm_names(config)
    metrics_client.put_metric_alarm(
        AlarmName=failure_alarm,
        AlarmDescription="At least one voice call failed in five minutes",
        ActionsEnabled=False,
        Namespace=config.metric_namespace,
        MetricName="CallsFailed",
        Dimensions=_dimensions(config.environment, "Call", outcome="FAILED"),
        Statistic="Sum",
        Period=300,
        EvaluationPeriods=1,
        DatapointsToAlarm=1,
        Threshold=1.0,
        ComparisonOperator="GreaterThanOrEqualToThreshold",
        TreatMissingData="notBreaching",
        Unit="Count",
    )
    metrics_client.put_metric_alarm(
        AlarmName=first_audio_alarm,
        AlarmDescription="Call start to first Nova audio exceeds three seconds",
        ActionsEnabled=False,
        Namespace=config.metric_namespace,
        MetricName="CallStartToFirstAudioMs",
        Dimensions=_dimensions(config.environment, "Call"),
        Statistic="Maximum",
        Period=300,
        EvaluationPeriods=1,
        DatapointsToAlarm=1,
        Threshold=3_000.0,
        ComparisonOperator="GreaterThanThreshold",
        TreatMissingData="notBreaching",
        Unit="Milliseconds",
    )
    response = metrics_client.describe_alarms(AlarmNames=[failure_alarm, first_audio_alarm])
    alarms = response.get("MetricAlarms", [])
    alarm_count = len(alarms) if isinstance(alarms, list) else 0
    dashboard = build_reliability_dashboard(config)
    name = dashboard_name(config)
    dashboard_body = json.dumps(dashboard, separators=(",", ":"), sort_keys=True)
    dashboard_response = metrics_client.put_dashboard(
        DashboardName=name,
        DashboardBody=dashboard_body,
    )
    validation_messages = dashboard_response.get("DashboardValidationMessages", [])
    if isinstance(validation_messages, list) and validation_messages:
        raise ValueError("CloudWatch rejected the reliability dashboard")
    inspected_dashboard = metrics_client.get_dashboard(DashboardName=name)
    inspected_body = inspected_dashboard.get("DashboardBody")
    if not isinstance(inspected_body, str) or json.loads(inspected_body) != dashboard:
        raise RuntimeError("CloudWatch dashboard inspection did not match the requested body")
    return BootstrapResult(
        log_group=config.log_group,
        log_stream=config.log_stream,
        retention_days=7,
        metric_namespace=config.metric_namespace,
        failure_alarm=failure_alarm,
        first_audio_alarm=first_audio_alarm,
        inspected_alarm_count=alarm_count,
        dashboard_name=name,
        dashboard_widget_count=len(cast(list[object], dashboard["widgets"])),
    )


def alarm_names(config: ObservabilityRuntimeConfig) -> tuple[str, str]:
    """Return stable environment-scoped demo alarm names."""
    prefix = f"{config.service_name}-{config.environment}"
    return f"{prefix}-call-failures", f"{prefix}-call-start-to-first-audio"


def dashboard_name(config: ObservabilityRuntimeConfig) -> str:
    """Return the stable environment-scoped reliability dashboard name."""
    return f"{config.service_name}-{config.environment}-reliability"


def build_reliability_dashboard(config: ObservabilityRuntimeConfig) -> dict[str, object]:
    """Build canonical low-cardinality CloudWatch dashboard JSON."""
    namespace = config.metric_namespace
    environment = config.environment

    def metric(
        name: str,
        component: str,
        *,
        outcome: str | None = None,
        stat: str = "Sum",
    ) -> list[object]:
        row: list[object] = [
            namespace,
            name,
            "Environment",
            environment,
            "Component",
            component,
        ]
        if outcome is not None:
            row.extend(("Outcome", outcome))
        row.append({"stat": stat})
        return row

    def widget(
        *,
        title: str,
        x: int,
        y: int,
        metrics: list[list[object]],
    ) -> dict[str, object]:
        return {
            "type": "metric",
            "x": x,
            "y": y,
            "width": 12,
            "height": 6,
            "properties": {
                "title": title,
                "region": config.region,
                "period": 300,
                "view": "timeSeries",
                "stacked": False,
                "metrics": metrics,
            },
        }

    return {
        "start": "-PT3H",
        "periodOverride": "inherit",
        "widgets": [
            widget(
                title="Call outcomes",
                x=0,
                y=0,
                metrics=[
                    metric("CallsStarted", "Call"),
                    metric("CallsCompleted", "Call", outcome="SUCCEEDED"),
                    metric("CallsCompleted", "Call", outcome="DISCONNECTED"),
                    metric("CallsFailed", "Call", outcome="FAILED"),
                ],
            ),
            widget(
                title="Call start to first audio and duration",
                x=12,
                y=0,
                metrics=[
                    metric("CallStartToFirstAudioMs", "Call", stat="Maximum"),
                    metric("CallDurationMs", "Call", outcome="SUCCEEDED", stat="Average"),
                    metric("CallDurationMs", "Call", outcome="FAILED", stat="Average"),
                ],
            ),
            widget(
                title="Component errors and persistence retries",
                x=0,
                y=6,
                metrics=[
                    metric("NovaErrors", "Nova"),
                    metric("TwilioErrors", "Twilio"),
                    metric("PersistenceErrors", "Persistence"),
                    metric("PersistenceRetries", "Persistence"),
                ],
            ),
            widget(
                title="Nova continuation health",
                x=12,
                y=6,
                metrics=[
                    metric("ContinuationAttempts", "Nova"),
                    metric("ContinuationSuccesses", "Nova"),
                    metric("ContinuationFailures", "Nova"),
                ],
            ),
        ],
    }


def run(argv: Sequence[str] | None = None) -> int:
    """Run the local observability administration command."""
    parser = argparse.ArgumentParser(prog="realtime-voice-observability")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("bootstrap", help="create/update CloudWatch logs and alarms")
    args = parser.parse_args(argv)
    if args.command != "bootstrap":
        parser.error("unsupported command")
    try:
        config = AppSettings().to_observability_runtime_config()
        session = boto3.Session(profile_name=config.profile, region_name=config.region)
        logs_client = cast(BootstrapLogsClient, cast(Any, session.client("logs")))
        metrics_client = cast(BootstrapMetricsClient, cast(Any, session.client("cloudwatch")))
        result = bootstrap_observability(
            config=config,
            logs_client=logs_client,
            metrics_client=metrics_client,
        )
    except (ConfigurationError, ValidationError, ValueError) as error:
        print(
            json.dumps(
                {"status": "error", "error_type": type(error).__name__},
                separators=(",", ":"),
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    except Exception as error:
        print(
            json.dumps(
                {"status": "error", "error_type": type(error).__name__},
                separators=(",", ":"),
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1
    print(json.dumps({"status": "ok", **asdict(result)}, sort_keys=True))
    return 0


def _create_log_group_if_needed(client: BootstrapLogsClient, log_group: str) -> None:
    try:
        client.create_log_group(logGroupName=log_group)
    except Exception as error:
        if _aws_error_code(error) != "ResourceAlreadyExistsException":
            raise


def _create_log_stream_if_needed(
    client: BootstrapLogsClient,
    log_group: str,
    log_stream: str,
) -> None:
    try:
        client.create_log_stream(logGroupName=log_group, logStreamName=log_stream)
    except Exception as error:
        if _aws_error_code(error) != "ResourceAlreadyExistsException":
            raise


def _dimensions(
    environment: str,
    component: str,
    *,
    outcome: str | None = None,
) -> list[dict[str, str]]:
    dimensions = [
        {"Name": "Environment", "Value": environment},
        {"Name": "Component", "Value": component},
    ]
    if outcome is not None:
        dimensions.append({"Name": "Outcome", "Value": outcome})
    return dimensions


def _aws_error_code(error: Exception) -> str | None:
    response = getattr(error, "response", None)
    if not isinstance(response, dict):
        return None
    details = response.get("Error")
    if not isinstance(details, dict):
        return None
    code = details.get("Code")
    return code if isinstance(code, str) else None
