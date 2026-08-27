"""Pure Nova 2 Sonic event construction, parsing, and lifecycle rules."""

from __future__ import annotations

import base64
import json
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Final, Literal, cast

from realtime_voice_agent.config import NovaRuntimeConfig
from realtime_voice_agent.transcript import (
    ConversationHistoryTurn,
    FinalTranscript,
    TranscriptSpeaker,
)

MEDIA_TYPE_PCM: Final = "audio/lpcm"
MEDIA_TYPE_TEXT: Final = "text/plain"


class NovaProtocolError(ValueError):
    """Raised when a Nova protocol event violates the local contract."""


class NovaSessionState(StrEnum):
    """Lifecycle states for one standalone Nova stream."""

    NEW = "NEW"
    STARTING = "STARTING"
    ACTIVE = "ACTIVE"
    CLOSING = "CLOSING"
    CLOSED = "CLOSED"
    FAILED = "FAILED"


_ALLOWED_TRANSITIONS: Final[dict[NovaSessionState, frozenset[NovaSessionState]]] = {
    NovaSessionState.NEW: frozenset({NovaSessionState.STARTING, NovaSessionState.CLOSED}),
    NovaSessionState.STARTING: frozenset(
        {NovaSessionState.ACTIVE, NovaSessionState.CLOSING, NovaSessionState.FAILED}
    ),
    NovaSessionState.ACTIVE: frozenset({NovaSessionState.CLOSING, NovaSessionState.FAILED}),
    NovaSessionState.CLOSING: frozenset({NovaSessionState.CLOSED, NovaSessionState.FAILED}),
    NovaSessionState.FAILED: frozenset({NovaSessionState.CLOSING, NovaSessionState.CLOSED}),
    NovaSessionState.CLOSED: frozenset(),
}


@dataclass(frozen=True, slots=True)
class NovaEventIds:
    """Names that correlate one Nova prompt and its content blocks."""

    prompt_name: str
    system_content_name: str
    audio_content_name: str


@dataclass(frozen=True, slots=True)
class OutputAudio:
    """Validated PCM16LE mono 24 kHz audio emitted by Nova."""

    pcm16le_24khz: bytes


@dataclass(frozen=True, slots=True)
class CompletionEnded:
    """Nova reported completion of the current response sequence."""


@dataclass(frozen=True, slots=True)
class InterruptionStarted:
    """Nova stopped the current response because the caller began speaking."""

    source_event_id: str


@dataclass(frozen=True, slots=True)
class ContinuationStarted:
    """A replacement Nova stream is starting before the current stream retires."""

    generation: int
    history_turns: int


@dataclass(frozen=True, slots=True)
class ContinuationSucceeded:
    """Input and output ownership moved atomically to a replacement stream."""

    generation: int
    history_turns: int
    buffered_pcm16_bytes: int
    retired_cleanup_failed: bool


@dataclass(frozen=True, slots=True)
class ContinuationFailed:
    """A replacement could not become the active stream."""

    generation: int
    phase: Literal["STARTUP", "HANDOFF"]


type NovaServerEvent = (
    OutputAudio
    | FinalTranscript
    | InterruptionStarted
    | CompletionEnded
    | ContinuationStarted
    | ContinuationSucceeded
    | ContinuationFailed
)


def transition_state(
    current: NovaSessionState,
    target: NovaSessionState,
) -> NovaSessionState:
    """Return a valid target state or reject the transition."""
    if target not in _ALLOWED_TRANSITIONS[current]:
        raise NovaProtocolError(f"invalid Nova session transition: {current} -> {target}")
    return target


def build_initialization_events(
    *,
    ids: NovaEventIds,
    config: NovaRuntimeConfig,
    system_prompt: str,
) -> tuple[bytes, ...]:
    """Build the official session/prompt/system event sequence."""
    if not system_prompt.strip():
        raise NovaProtocolError("system prompt must not be blank")

    return (
        _encode_event(
            "sessionStart",
            {
                "inferenceConfiguration": {
                    "maxTokens": 1024,
                    "topP": 0.9,
                    "temperature": 0.7,
                },
                "turnDetectionConfiguration": {"endpointingSensitivity": "HIGH"},
            },
        ),
        _encode_event(
            "promptStart",
            {
                "promptName": ids.prompt_name,
                "textOutputConfiguration": {"mediaType": MEDIA_TYPE_TEXT},
                "audioOutputConfiguration": {
                    "mediaType": MEDIA_TYPE_PCM,
                    "sampleRateHertz": config.output_sample_rate,
                    "sampleSizeBits": config.sample_width_bytes * 8,
                    "channelCount": config.channels,
                    "voiceId": config.voice_id,
                    "encoding": "base64",
                    "audioType": "SPEECH",
                },
            },
        ),
        _encode_event(
            "contentStart",
            {
                "promptName": ids.prompt_name,
                "contentName": ids.system_content_name,
                "type": "TEXT",
                "interactive": True,
                "role": "SYSTEM",
                "textInputConfiguration": {"mediaType": MEDIA_TYPE_TEXT},
            },
        ),
        _encode_event(
            "textInput",
            {
                "promptName": ids.prompt_name,
                "contentName": ids.system_content_name,
                "content": system_prompt,
            },
        ),
        build_content_end(
            prompt_name=ids.prompt_name,
            content_name=ids.system_content_name,
        ),
    )


def build_history_events(
    *,
    prompt_name: str,
    history: tuple[ConversationHistoryTurn, ...],
) -> tuple[bytes, ...]:
    """Build official non-interactive USER/ASSISTANT text blocks for continuation."""
    events: list[bytes] = []
    for turn in history:
        content_name = str(uuid.uuid5(uuid.NAMESPACE_URL, f"nova-history:{turn.source_event_id}"))
        events.extend(
            (
                _encode_event(
                    "contentStart",
                    {
                        "promptName": prompt_name,
                        "contentName": content_name,
                        "type": "TEXT",
                        "interactive": False,
                        "role": turn.role.value,
                        "textInputConfiguration": {"mediaType": MEDIA_TYPE_TEXT},
                    },
                ),
                _encode_event(
                    "textInput",
                    {
                        "promptName": prompt_name,
                        "contentName": content_name,
                        "content": turn.text,
                    },
                ),
                build_content_end(
                    prompt_name=prompt_name,
                    content_name=content_name,
                ),
            )
        )
    return tuple(events)


def build_interactive_text_events(
    *,
    prompt_name: str,
    text: str,
) -> tuple[bytes, ...]:
    """Build one interactive USER text turn that prompts a spoken response."""
    if not prompt_name.strip() or not text.strip():
        raise NovaProtocolError("interactive text prompt values must not be blank")
    content_name = str(uuid.uuid4())
    return (
        _encode_event(
            "contentStart",
            {
                "promptName": prompt_name,
                "contentName": content_name,
                "type": "TEXT",
                "interactive": True,
                "role": "USER",
                "textInputConfiguration": {"mediaType": MEDIA_TYPE_TEXT},
            },
        ),
        _encode_event(
            "textInput",
            {
                "promptName": prompt_name,
                "contentName": content_name,
                "content": text,
            },
        ),
        build_content_end(
            prompt_name=prompt_name,
            content_name=content_name,
        ),
    )


def build_audio_content_start(
    *,
    ids: NovaEventIds,
    config: NovaRuntimeConfig,
) -> bytes:
    """Build the explicit PCM16LE mono 16 kHz input contract event."""
    return _encode_event(
        "contentStart",
        {
            "promptName": ids.prompt_name,
            "contentName": ids.audio_content_name,
            "type": "AUDIO",
            "interactive": True,
            "role": "USER",
            "audioInputConfiguration": {
                "mediaType": MEDIA_TYPE_PCM,
                "sampleRateHertz": config.input_sample_rate,
                "sampleSizeBits": config.sample_width_bytes * 8,
                "channelCount": config.channels,
                "audioType": "SPEECH",
                "encoding": "base64",
            },
        },
    )


def build_audio_input(*, ids: NovaEventIds, pcm16le_16khz: bytes) -> bytes:
    """Validate and encode one raw microphone chunk for Nova."""
    validate_pcm16_chunk(pcm16le_16khz)
    content = base64.b64encode(pcm16le_16khz).decode("ascii")
    return _encode_event(
        "audioInput",
        {
            "promptName": ids.prompt_name,
            "contentName": ids.audio_content_name,
            "content": content,
        },
    )


def build_content_end(*, prompt_name: str, content_name: str) -> bytes:
    """Build the common content-end event."""
    return _encode_event(
        "contentEnd",
        {"promptName": prompt_name, "contentName": content_name},
    )


def build_prompt_end(*, prompt_name: str) -> bytes:
    """Build the prompt-end event."""
    return _encode_event("promptEnd", {"promptName": prompt_name})


def build_session_end() -> bytes:
    """Build the terminal session-end event."""
    return _encode_event("sessionEnd", {})


@dataclass(slots=True)
class _TranscriptBuffer:
    speaker: TranscriptSpeaker
    parts: list[str]


class NovaEventParser:
    """Parse Nova output while assembling only FINAL text content blocks."""

    def __init__(self) -> None:
        self._final_transcripts: dict[str, _TranscriptBuffer] = {}

    def parse(self, payload: bytes) -> NovaServerEvent | None:
        """Return one stable domain event or ignore an unsupported protocol event."""
        event = _decode_event(payload)

        audio_output = event.get("audioOutput")
        if audio_output is not None:
            if not isinstance(audio_output, dict):
                raise NovaProtocolError("Nova audioOutput must be an object")
            content = audio_output.get("content")
            if not isinstance(content, str):
                raise NovaProtocolError("Nova audioOutput is missing base64 content")
            try:
                audio = base64.b64decode(content, validate=True)
            except ValueError as error:
                raise NovaProtocolError("Nova audioOutput content is not valid base64") from error
            validate_pcm16_chunk(audio)
            return OutputAudio(pcm16le_24khz=audio)

        content_start = event.get("contentStart")
        if content_start is not None:
            self._handle_content_start(content_start)
            return None

        text_output = event.get("textOutput")
        if text_output is not None:
            self._handle_text_output(text_output)
            return None

        content_end = event.get("contentEnd")
        if content_end is not None:
            return self._handle_content_end(content_end)

        if "completionEnd" in event:
            return CompletionEnded()

        return None

    def _handle_content_start(self, value: object) -> None:
        if not isinstance(value, dict):
            raise NovaProtocolError("Nova contentStart must be an object")
        if value.get("type") != "TEXT":
            return
        content_id = _required_event_str(value, "contentId", "contentStart")
        role = _required_event_str(value, "role", "contentStart")
        additional_fields = value.get("additionalModelFields")
        if not isinstance(additional_fields, str):
            return
        try:
            fields = json.loads(additional_fields)
        except json.JSONDecodeError as error:
            raise NovaProtocolError(
                "Nova contentStart additionalModelFields is not valid JSON"
            ) from error
        if not isinstance(fields, dict) or fields.get("generationStage") != "FINAL":
            return
        if role == "USER":
            speaker = TranscriptSpeaker.CALLER
        elif role == "ASSISTANT":
            speaker = TranscriptSpeaker.AGENT
        else:
            raise NovaProtocolError(f"Unsupported final transcript role: {role}")
        self._final_transcripts[content_id] = _TranscriptBuffer(speaker=speaker, parts=[])

    def _handle_text_output(self, value: object) -> None:
        if not isinstance(value, dict):
            raise NovaProtocolError("Nova textOutput must be an object")
        content_id = _required_event_str(value, "contentId", "textOutput")
        content = _required_event_str(value, "content", "textOutput", allow_blank=True)
        transcript = self._final_transcripts.get(content_id)
        if transcript is not None:
            transcript.parts.append(content)

    def _handle_content_end(
        self,
        value: object,
    ) -> FinalTranscript | InterruptionStarted | None:
        if not isinstance(value, dict):
            raise NovaProtocolError("Nova contentEnd must be an object")
        if value.get("type") != "TEXT":
            return None
        content_id = _required_event_str(value, "contentId", "contentEnd")
        transcript = self._final_transcripts.pop(content_id, None)
        if value.get("stopReason") == "INTERRUPTED":
            return InterruptionStarted(source_event_id=content_id)
        if transcript is None:
            return None
        text = "".join(transcript.parts).strip()
        if not text or _is_interruption_control(text):
            return None
        return FinalTranscript(
            speaker=transcript.speaker,
            text=text,
            source_event_id=content_id,
        )


def _is_interruption_control(text: str) -> bool:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return False
    return (
        isinstance(value, dict)
        and set(value) == {"interrupted"}
        and isinstance(value["interrupted"], bool)
    )


def _decode_event(payload: bytes) -> dict[str, object]:
    try:
        document = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise NovaProtocolError("Nova response was not valid UTF-8 JSON") from error
    if not isinstance(document, dict):
        raise NovaProtocolError("Nova response must be a JSON object")
    event = document.get("event")
    if not isinstance(event, dict):
        raise NovaProtocolError("Nova response is missing its event object")
    return cast(dict[str, object], event)


def _required_event_str(
    value: dict[object, object],
    field: str,
    event_name: str,
    *,
    allow_blank: bool = False,
) -> str:
    parsed = value.get(field)
    if not isinstance(parsed, str) or (not allow_blank and not parsed):
        raise NovaProtocolError(f"Nova {event_name} is missing {field}")
    return parsed


def validate_pcm16_chunk(audio: bytes) -> None:
    """Validate raw mono PCM16 chunk alignment without copying it."""
    if not audio:
        raise NovaProtocolError("PCM16 audio chunk must not be empty")
    if len(audio) % 2 != 0:
        raise NovaProtocolError("PCM16 audio chunk must contain complete 16-bit samples")


def _encode_event(name: str, body: Mapping[str, object]) -> bytes:
    document = {"event": {name: cast(dict[str, object], dict(body))}}
    return json.dumps(document, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
