"""Narrow synchronous repository boundaries isolated behind async workers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from realtime_voice_agent.persistence.models import (
    Persona,
    SessionStart,
    SessionTerminal,
    TranscriptTurn,
    TranscriptView,
)


class PersonaRepository(Protocol):
    """Versioned persona administration and active-persona lookup."""

    def list_personas(self) -> Sequence[Persona]: ...

    def get_persona(self, persona_id: str) -> Persona | None: ...

    def get_active_persona(self) -> Persona: ...

    def put_persona(
        self,
        *,
        persona_id: str,
        name: str,
        system_prompt: str,
        voice_id: str,
        expected_version: int,
    ) -> Persona: ...

    def activate_persona(self, persona_id: str, *, expected_version: int) -> Persona: ...


class SessionRepository(Protocol):
    """Write-once call data and deterministic transcript retrieval."""

    def create_session(self, session: SessionStart) -> None: ...

    def mark_session_active(self, session_id: str, activated_at: str) -> None: ...

    def append_transcript_turn(self, turn: TranscriptTurn) -> bool: ...

    def finish_session(self, terminal: SessionTerminal) -> bool: ...

    def get_transcript(
        self,
        *,
        session_id: str | None = None,
        call_sid: str | None = None,
    ) -> TranscriptView: ...


class PersistenceStore(PersonaRepository, SessionRepository, Protocol):
    """Combined runtime persistence contract implemented by one DynamoDB store."""
