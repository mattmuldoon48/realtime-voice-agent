"""Immutable persona, session, and transcript persistence values."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from realtime_voice_agent.transcript import TranscriptSpeaker


class PersistedSessionStatus(StrEnum):
    """Session lifecycle values written to DynamoDB."""

    STARTING = "STARTING"
    ACTIVE = "ACTIVE"
    COMPLETED = "COMPLETED"
    DISCONNECTED = "DISCONNECTED"
    FAILED = "FAILED"


@dataclass(frozen=True, slots=True)
class Persona:
    """Versioned voice-agent behavior configured outside a deployment."""

    persona_id: str
    name: str
    system_prompt: str
    voice_id: str
    version: int
    active: bool
    created_at: datetime
    updated_at: datetime

    def __post_init__(self) -> None:
        _require_non_blank("persona_id", self.persona_id)
        _require_non_blank("name", self.name)
        _require_non_blank("system_prompt", self.system_prompt)
        _require_non_blank("voice_id", self.voice_id)
        if self.version <= 0:
            raise ValueError("persona version must be positive")
        _require_aware("created_at", self.created_at)
        _require_aware("updated_at", self.updated_at)

    def snapshot(self) -> PersonaSnapshot:
        """Capture the immutable behavior used by one call."""
        return PersonaSnapshot(
            persona_id=self.persona_id,
            name=self.name,
            system_prompt=self.system_prompt,
            voice_id=self.voice_id,
            version=self.version,
        )


@dataclass(frozen=True, slots=True)
class PersonaSnapshot:
    """Persona values frozen into a call when its WebSocket connects."""

    persona_id: str
    name: str
    system_prompt: str
    voice_id: str
    version: int

    def __post_init__(self) -> None:
        _require_non_blank("persona_id", self.persona_id)
        _require_non_blank("name", self.name)
        _require_non_blank("system_prompt", self.system_prompt)
        _require_non_blank("voice_id", self.voice_id)
        if self.version <= 0:
            raise ValueError("persona version must be positive")


@dataclass(frozen=True, slots=True)
class SessionStart:
    """Metadata written before a call becomes active."""

    session_id: str
    call_sid: str
    stream_sid: str
    persona: PersonaSnapshot
    model_id: str
    started_at: datetime
    expires_at: int


@dataclass(frozen=True, slots=True)
class SessionTerminal:
    """Final lifecycle update written exactly once for one call."""

    session_id: str
    status: PersistedSessionStatus
    outcome: str
    termination_reason: str
    ended_at: datetime
    duration_ms: int
    inbound_media_frames: int
    nova_input_frames: int
    nova_input_pcm16_bytes: int
    nova_response_audio_events: int
    outbound_media_frames: int
    outbound_mulaw_bytes: int
    caller_turns: int
    agent_turns: int
    failure_code: str | None
    cleanup_error_code: str | None


@dataclass(frozen=True, slots=True)
class TranscriptTurn:
    """One ordered final caller or agent transcript turn."""

    session_id: str
    turn_number: int
    speaker: TranscriptSpeaker
    text: str
    source_event_id: str
    created_at: datetime
    expires_at: int

    def __post_init__(self) -> None:
        if self.turn_number <= 0:
            raise ValueError("turn number must be positive")
        _require_non_blank("text", self.text)
        _require_non_blank("source_event_id", self.source_event_id)
        _require_aware("created_at", self.created_at)


@dataclass(frozen=True, slots=True)
class TranscriptView:
    """Session metadata and its final transcript turns in numeric order."""

    session: dict[str, object]
    turns: tuple[TranscriptTurn, ...]


def _require_non_blank(field: str, value: str) -> None:
    if not value.strip():
        raise ValueError(f"{field} must not be blank")


def _require_aware(field: str, value: datetime) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")
