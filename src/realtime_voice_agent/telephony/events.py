"""Typed Twilio Media Streams protocol parsing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from realtime_voice_agent.audio.codecs import TWILIO_CHANNELS, TWILIO_ENCODING, TWILIO_SAMPLE_RATE

TwilioEventName = Literal["connected", "start", "media", "mark", "stop"]


class TwilioProtocolError(ValueError):
    """Raised for malformed or unsupported Twilio Media Streams events."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class ConnectedEvent:
    sequence_number: int | None
    protocol: str | None
    version: str | None


@dataclass(frozen=True, slots=True)
class MediaFormat:
    encoding: str
    sample_rate: int
    channels: int


@dataclass(frozen=True, slots=True)
class StartEvent:
    sequence_number: int
    stream_sid: str
    call_sid: str
    account_sid: str
    media_format: MediaFormat
    custom_parameters: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class MediaEvent:
    sequence_number: int
    chunk: int
    timestamp_ms: int
    payload: str
    track: str | None


@dataclass(frozen=True, slots=True)
class MarkEvent:
    sequence_number: int
    name: str


@dataclass(frozen=True, slots=True)
class StopEvent:
    sequence_number: int


TwilioMediaStreamEvent = ConnectedEvent | StartEvent | MediaEvent | MarkEvent | StopEvent


@dataclass(frozen=True, slots=True)
class TwilioMediaCommand:
    """One bounded outbound raw mu-law media frame."""

    stream_sid: str
    payload: str
    mulaw_bytes: int
    generation: int

    def to_json(self) -> dict[str, object]:
        return {
            "event": "media",
            "streamSid": self.stream_sid,
            "media": {"payload": self.payload},
        }


@dataclass(frozen=True, slots=True)
class TwilioMarkCommand:
    """One response-level playback marker."""

    stream_sid: str
    name: str
    generation: int

    def to_json(self) -> dict[str, object]:
        return {
            "event": "mark",
            "streamSid": self.stream_sid,
            "mark": {"name": self.name},
        }


@dataclass(frozen=True, slots=True)
class TwilioClearCommand:
    """Clear Twilio's buffered outbound audio."""

    stream_sid: str
    generation: int

    def to_json(self) -> dict[str, object]:
        return {
            "event": "clear",
            "streamSid": self.stream_sid,
        }


type TwilioOutboundCommand = TwilioMediaCommand | TwilioMarkCommand | TwilioClearCommand


def parse_twilio_event(message: object) -> TwilioMediaStreamEvent:
    """Parse a Twilio WebSocket JSON object without retaining unbounded external fields."""
    if not isinstance(message, dict):
        raise TwilioProtocolError("TWILIO_EVENT_NOT_OBJECT", "Twilio event must be a JSON object")

    event_name = _required_str(message, "event")
    if event_name == "connected":
        return ConnectedEvent(
            sequence_number=_optional_int(message.get("sequenceNumber")),
            protocol=_optional_str(message.get("protocol")),
            version=_optional_str(message.get("version")),
        )
    if event_name == "start":
        return _parse_start(message)
    if event_name == "media":
        return _parse_media(message)
    if event_name == "mark":
        return _parse_mark(message)
    if event_name == "stop":
        return StopEvent(sequence_number=_required_sequence(message))
    raise TwilioProtocolError("TWILIO_UNKNOWN_EVENT", "Unsupported Twilio event type")


def validate_media_format(media_format: MediaFormat) -> None:
    """Enforce Twilio's required raw mu-law 8 kHz mono stream contract."""
    if media_format.encoding != TWILIO_ENCODING:
        raise TwilioProtocolError("TWILIO_MEDIA_ENCODING_UNSUPPORTED", "Unsupported media encoding")
    if media_format.sample_rate != TWILIO_SAMPLE_RATE:
        raise TwilioProtocolError("TWILIO_MEDIA_SAMPLE_RATE_UNSUPPORTED", "Unsupported sample rate")
    if media_format.channels != TWILIO_CHANNELS:
        raise TwilioProtocolError("TWILIO_MEDIA_CHANNELS_UNSUPPORTED", "Unsupported channel count")


def _parse_start(message: dict[object, object]) -> StartEvent:
    start = _required_object(message, "start")
    media_format = _required_object(start, "mediaFormat")
    event = StartEvent(
        sequence_number=_required_sequence(message),
        stream_sid=_required_str(start, "streamSid"),
        call_sid=_required_str(start, "callSid"),
        account_sid=_required_str(start, "accountSid"),
        media_format=MediaFormat(
            encoding=_required_str(media_format, "encoding"),
            sample_rate=_required_int(media_format, "sampleRate"),
            channels=_required_int(media_format, "channels"),
        ),
        custom_parameters=_optional_string_pairs(start.get("customParameters")),
    )
    validate_media_format(event.media_format)
    return event


def _parse_media(message: dict[object, object]) -> MediaEvent:
    media = _required_object(message, "media")
    return MediaEvent(
        sequence_number=_required_sequence(message),
        chunk=_required_int(media, "chunk"),
        timestamp_ms=_required_int(media, "timestamp"),
        payload=_required_str(media, "payload"),
        track=_optional_str(media.get("track")),
    )


def _parse_mark(message: dict[object, object]) -> MarkEvent:
    mark = _required_object(message, "mark")
    return MarkEvent(sequence_number=_required_sequence(message), name=_required_str(mark, "name"))


def _required_sequence(message: dict[object, object]) -> int:
    return _required_int(message, "sequenceNumber")


def _required_object(message: dict[object, object], field: str) -> dict[object, object]:
    value = message.get(field)
    if not isinstance(value, dict):
        raise TwilioProtocolError("TWILIO_FIELD_MISSING", f"Twilio field is missing: {field}")
    return value


def _required_str(message: dict[object, object], field: str) -> str:
    value = message.get(field)
    if not isinstance(value, str) or not value:
        raise TwilioProtocolError("TWILIO_FIELD_MISSING", f"Twilio field is missing: {field}")
    return value


def _required_int(message: dict[object, object], field: str) -> int:
    value = message.get(field)
    parsed = _optional_int(value)
    if parsed is None:
        raise TwilioProtocolError("TWILIO_FIELD_MISSING", f"Twilio field is missing: {field}")
    return parsed


def _optional_str(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _optional_string_pairs(value: object) -> tuple[tuple[str, str], ...]:
    if value is None:
        return ()
    if not isinstance(value, dict) or len(value) > 16:
        raise TwilioProtocolError(
            "TWILIO_CUSTOM_PARAMETERS_INVALID",
            "Twilio custom parameters are invalid",
        )
    parameters: list[tuple[str, str]] = []
    for key, item in value.items():
        if (
            not isinstance(key, str)
            or not key
            or len(key) > 64
            or not isinstance(item, str)
            or not item
            or len(item) > 256
        ):
            raise TwilioProtocolError(
                "TWILIO_CUSTOM_PARAMETERS_INVALID",
                "Twilio custom parameters are invalid",
            )
        parameters.append((key, item))
    return tuple(parameters)


def _optional_int(value: object) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdecimal():
        return int(value)
    return None
