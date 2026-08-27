"""Conversation transcript domain values."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class TranscriptSpeaker(StrEnum):
    """Bounded speaker labels persisted for final conversation turns."""

    CALLER = "CALLER"
    AGENT = "AGENT"


class ConversationRole(StrEnum):
    """Nova roles accepted for non-interactive conversation history."""

    USER = "USER"
    ASSISTANT = "ASSISTANT"


@dataclass(frozen=True, slots=True)
class FinalTranscript:
    """One final transcript emitted by Nova after a content block ends."""

    speaker: TranscriptSpeaker
    text: str
    source_event_id: str

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("final transcript text must not be blank")
        if not self.source_event_id.strip():
            raise ValueError("final transcript source event ID must not be blank")


@dataclass(frozen=True, slots=True)
class ConversationHistoryTurn:
    """One immutable, bounded FINAL turn replayed into a replacement Nova stream."""

    role: ConversationRole
    text: str
    source_event_id: str

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("conversation history text must not be blank")
        if not self.source_event_id.strip():
            raise ValueError("conversation history source event ID must not be blank")

    @classmethod
    def from_final_transcript(cls, transcript: FinalTranscript) -> ConversationHistoryTurn:
        """Map one final caller/agent transcript to Nova's history role."""
        role = (
            ConversationRole.USER
            if transcript.speaker is TranscriptSpeaker.CALLER
            else ConversationRole.ASSISTANT
        )
        return cls(
            role=role,
            text=transcript.text,
            source_event_id=transcript.source_event_id,
        )
