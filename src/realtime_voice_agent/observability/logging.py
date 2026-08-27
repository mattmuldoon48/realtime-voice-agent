"""Structured local logging that is safe for the real-time media path."""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from typing import Final, cast

import structlog
from structlog.typing import (
    EventDict,
    FilteringBoundLogger,
    Processor,
    WrappedLogger,
)

_DEFAULT_SERVICE: Final = "realtime-voice-agent"
_REDACTED: Final = "[REDACTED]"
_SENSITIVE_KEY_PARTS: Final = (
    "authorization",
    "credential",
    "phone",
    "prompt",
    "raw_audio",
    "secret",
    "token",
    "transcript",
)
_SENSITIVE_EXACT_KEYS: Final = frozenset({"audio", "headers", "payload", "text"})


type CloudLogSink = Callable[[dict[str, object]], bool]


def configure_local_logging(
    *,
    level: str,
    service: str = _DEFAULT_SERVICE,
    cloud_log_sink: CloudLogSink | None = None,
) -> None:
    """Configure sanitized local JSON and optional non-blocking CloudWatch delivery."""
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError("LOG_LEVEL must be a standard Python logging level")

    processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        _sanitize_processor,
    ]
    if cloud_log_sink is not None:
        processors.append(_cloud_sink_processor(cloud_log_sink))
    processors.append(structlog.processors.JSONRenderer())
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(*, service: str = _DEFAULT_SERVICE) -> FilteringBoundLogger:
    """Return a logger with bounded service context."""
    return cast(FilteringBoundLogger, structlog.get_logger(service=service))


def sanitize_event(event: Mapping[str, object]) -> dict[str, object]:
    """Return a recursively sanitized copy of one structured log event."""
    return {key: _sanitize_value(key, value) for key, value in event.items()}


def emit_local_event(
    event: str,
    *,
    level: str,
    service: str = _DEFAULT_SERVICE,
    **fields: object,
) -> None:
    """Write one local JSON event directly, bypassing CloudWatch processors."""
    document = sanitize_event(
        {
            "service": service,
            "event": event,
            "level": level,
            "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            **fields,
        }
    )
    print(
        json.dumps(document, default=str, separators=(",", ":"), sort_keys=True),
        file=sys.stdout,
        flush=True,
    )


def _sanitize_processor(
    _logger: WrappedLogger,
    _method_name: str,
    event_dict: EventDict,
) -> EventDict:
    return cast(EventDict, sanitize_event(cast(Mapping[str, object], event_dict)))


def _cloud_sink_processor(cloud_log_sink: CloudLogSink) -> Processor:
    def publish(
        _logger: WrappedLogger,
        _method_name: str,
        event_dict: EventDict,
    ) -> EventDict:
        try:
            cloud_log_sink(dict(event_dict))
        except Exception as error:
            emit_local_event(
                "cloudwatch_log_enqueue_failed",
                level="error",
                error_type=type(error).__name__,
            )
        return event_dict

    return publish


def _sanitize_value(key: str, value: object) -> object:
    normalized_key = key.casefold()
    if _is_sensitive_key(normalized_key):
        return _REDACTED
    if isinstance(value, Mapping):
        return sanitize_event(cast(Mapping[str, object], value))
    if isinstance(value, (list, tuple)):
        return [
            sanitize_event(cast(Mapping[str, object], item)) if isinstance(item, Mapping) else item
            for item in value
        ]
    return value


def _is_sensitive_key(normalized_key: str) -> bool:
    if normalized_key.endswith(("_bytes", "_frames", "_events")):
        return False
    return normalized_key in _SENSITIVE_EXACT_KEYS or any(
        part in normalized_key for part in _SENSITIVE_KEY_PARTS
    )
