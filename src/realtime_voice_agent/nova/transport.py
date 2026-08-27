"""Application-facing boundary for Nova bidirectional streaming."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol

from realtime_voice_agent.nova.events import NovaServerEvent, NovaSessionState
from realtime_voice_agent.transcript import ConversationHistoryTurn


class NovaSonicTransport(Protocol):
    """Stable contract that contains the experimental AWS SDK."""

    @property
    def state(self) -> NovaSessionState: ...

    async def start(
        self,
        *,
        system_prompt: str,
        history: tuple[ConversationHistoryTurn, ...] = (),
    ) -> None: ...

    async def start_audio_input(self) -> None: ...

    async def send_audio(self, pcm16le_16khz: bytes) -> None: ...

    def events(self) -> AsyncIterator[NovaServerEvent]: ...

    async def finish_input(self) -> None: ...

    async def close(self) -> None: ...
