"""Bounded asynchronous CloudWatch Logs and custom-metric delivery."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol, cast

import boto3  # type: ignore[import-untyped]

from realtime_voice_agent.config import ObservabilityRuntimeConfig
from realtime_voice_agent.observability.logging import emit_local_event
from realtime_voice_agent.observability.models import MetricDatum, TelemetryPublisher


class TelemetryService(TelemetryPublisher, Protocol):
    """Application-lifetime telemetry publisher contract."""

    async def start(self) -> None:
        """Start background delivery."""

    async def close(self) -> None:
        """Flush and stop background delivery."""


class CloudWatchLogsClient(Protocol):
    """Narrow synchronous CloudWatch Logs client boundary."""

    def create_log_stream(self, **kwargs: object) -> Mapping[str, object]: ...

    def put_log_events(self, **kwargs: object) -> Mapping[str, object]: ...


class CloudWatchMetricsClient(Protocol):
    """Narrow synchronous CloudWatch metrics client boundary."""

    def put_metric_data(self, **kwargs: object) -> Mapping[str, object]: ...


@dataclass(frozen=True, slots=True)
class _QueuedLog:
    timestamp_ms: int
    message: str


type _QueueItem = _QueuedLog | MetricDatum | None


class CloudWatchTelemetry:
    """Queue telemetry synchronously and publish it from one background worker."""

    def __init__(
        self,
        *,
        config: ObservabilityRuntimeConfig,
        logs_client: CloudWatchLogsClient,
        metrics_client: CloudWatchMetricsClient,
        clock_ms: Callable[[], int] | None = None,
    ) -> None:
        self._config = config
        self._logs_client = logs_client
        self._metrics_client = metrics_client
        self._clock_ms = clock_ms or _epoch_milliseconds
        self._queue: asyncio.Queue[_QueueItem] = asyncio.Queue(maxsize=config.queue_max_events)
        self._worker: asyncio.Task[None] | None = None
        self._closed = False
        self._overflow_reported = False

    async def start(self) -> None:
        """Create the configured stream if needed and start the worker once."""
        if self._worker is not None:
            return
        await self._call_with_retry(self._create_log_stream, operation="CreateLogStream")
        self._worker = asyncio.create_task(self._run(), name="cloudwatch-telemetry-worker")

    def publish_log(self, event: dict[str, object]) -> bool:
        """Queue one sanitized JSON event without blocking its caller."""
        message = json.dumps(event, default=str, separators=(",", ":"), sort_keys=True)
        return self._put_nowait(_QueuedLog(timestamp_ms=self._clock_ms(), message=message))

    def publish_metric(self, metric: MetricDatum) -> bool:
        """Queue one validated metric without blocking its caller."""
        return self._put_nowait(metric)

    async def close(self) -> None:
        """Flush queued telemetry and stop the worker without raising into app shutdown."""
        if self._closed:
            return
        self._closed = True
        worker = self._worker
        if worker is None:
            return
        try:
            async with asyncio.timeout(self._config.cleanup_timeout_seconds):
                await self._queue.put(None)
                await asyncio.shield(worker)
        except TimeoutError:
            worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)
            emit_local_event(
                "telemetry_shutdown_timed_out",
                level="error",
                operation="Flush",
            )

    async def _run(self) -> None:
        while True:
            first = await self._queue.get()
            if first is None:
                return
            batch: list[_QueuedLog | MetricDatum] = [first]
            closing = False
            deadline = asyncio.get_running_loop().time() + self._config.flush_interval_seconds
            while len(batch) < self._config.batch_max_items:
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    break
                try:
                    item = await asyncio.wait_for(self._queue.get(), timeout=remaining)
                except TimeoutError:
                    break
                if item is None:
                    closing = True
                    break
                batch.append(item)
            await self._flush(batch)
            if closing:
                return

    async def _flush(self, batch: list[_QueuedLog | MetricDatum]) -> None:
        log_events = [item for item in batch if isinstance(item, _QueuedLog)]
        metrics = [item for item in batch if isinstance(item, MetricDatum)]
        if log_events:
            log_events.sort(key=lambda event: event.timestamp_ms)
            await self._call_with_retry(
                lambda: self._logs_client.put_log_events(
                    logGroupName=self._config.log_group,
                    logStreamName=self._config.log_stream,
                    logEvents=[
                        {"timestamp": event.timestamp_ms, "message": event.message}
                        for event in log_events
                    ],
                ),
                operation="PutLogEvents",
            )
        if metrics:
            await self._call_with_retry(
                lambda: self._metrics_client.put_metric_data(
                    Namespace=self._config.metric_namespace,
                    MetricData=[_metric_request(metric) for metric in metrics],
                ),
                operation="PutMetricData",
            )

    def _put_nowait(self, item: _QueuedLog | MetricDatum) -> bool:
        if self._closed:
            return False
        try:
            self._queue.put_nowait(item)
        except asyncio.QueueFull:
            if not self._overflow_reported:
                self._overflow_reported = True
                emit_local_event(
                    "telemetry_queue_overflow",
                    level="error",
                    queue_max_events=self._queue.maxsize,
                )
            return False
        return True

    def _create_log_stream(self) -> Mapping[str, object]:
        try:
            return self._logs_client.create_log_stream(
                logGroupName=self._config.log_group,
                logStreamName=self._config.log_stream,
            )
        except Exception as error:
            if _aws_error_code(error) == "ResourceAlreadyExistsException":
                return {}
            raise

    async def _call_with_retry(
        self,
        operation_call: Callable[[], object],
        *,
        operation: str,
    ) -> bool:
        for attempt in range(1, self._config.max_attempts + 1):
            try:
                await asyncio.to_thread(operation_call)
            except Exception as error:
                emit_local_event(
                    "cloudwatch_delivery_failed",
                    level="error",
                    operation=operation,
                    attempt=attempt,
                    max_attempts=self._config.max_attempts,
                    error_type=type(error).__name__,
                    error_code=_aws_error_code(error),
                )
                if attempt < self._config.max_attempts:
                    await asyncio.sleep(min(0.1 * (2 ** (attempt - 1)), 1.0))
            else:
                return True
        return False


def create_cloudwatch_telemetry(config: ObservabilityRuntimeConfig) -> CloudWatchTelemetry:
    """Construct CloudWatch clients through the standard AWS credential chain."""
    session = boto3.Session(profile_name=config.profile, region_name=config.region)
    logs_client = cast(CloudWatchLogsClient, cast(Any, session.client("logs")))
    metrics_client = cast(CloudWatchMetricsClient, cast(Any, session.client("cloudwatch")))
    return CloudWatchTelemetry(
        config=config,
        logs_client=logs_client,
        metrics_client=metrics_client,
    )


def _metric_request(metric: MetricDatum) -> dict[str, object]:
    return {
        "MetricName": metric.name.value,
        "Value": metric.value,
        "Unit": metric.unit,
        "Dimensions": [
            {"Name": dimension.name, "Value": dimension.value} for dimension in metric.dimensions
        ],
    }


def _epoch_milliseconds() -> int:
    return time.time_ns() // 1_000_000


def _aws_error_code(error: Exception) -> str | None:
    response = getattr(error, "response", None)
    if not isinstance(response, dict):
        return None
    error_details = response.get("Error")
    if not isinstance(error_details, dict):
        return None
    code = error_details.get("Code")
    return code if isinstance(code, str) else None
