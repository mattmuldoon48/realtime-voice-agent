"""Deterministic Nova connection rotation and history replay tests."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

import pytest

from realtime_voice_agent.nova.continuation import ContinuingNovaSonicTransport
from realtime_voice_agent.nova.events import (
    ContinuationFailed,
    ContinuationStarted,
    ContinuationSucceeded,
    NovaServerEvent,
    NovaSessionState,
    OutputAudio,
    build_history_events,
)
from realtime_voice_agent.transcript import (
    ConversationHistoryTurn,
    ConversationRole,
    FinalTranscript,
    TranscriptSpeaker,
)


class FakeTransport:
    def __init__(
        self,
        *,
        start_error: Exception | None = None,
        send_error: Exception | None = None,
        close_error: Exception | None = None,
        block_start: bool = False,
        block_close: bool = False,
    ) -> None:
        self._state = NovaSessionState.NEW
        self._start_error = start_error
        self._send_error = send_error
        self._close_error = close_error
        self._start_gate = asyncio.Event()
        self._close_gate = asyncio.Event()
        self._close_lock = asyncio.Lock()
        if not block_start:
            self._start_gate.set()
        if not block_close:
            self._close_gate.set()
        self._events: asyncio.Queue[NovaServerEvent | None] = asyncio.Queue()
        self.system_prompt: str | None = None
        self.history: tuple[ConversationHistoryTurn, ...] = ()
        self.sent_audio: list[bytes] = []
        self.sent_text: list[str] = []
        self.audio_starts = 0
        self.close_calls = 0
        self.start_started = asyncio.Event()
        self.close_started = asyncio.Event()

    @property
    def state(self) -> NovaSessionState:
        return self._state

    async def start(
        self,
        *,
        system_prompt: str,
        history: tuple[ConversationHistoryTurn, ...] = (),
    ) -> None:
        self.start_started.set()
        await self._start_gate.wait()
        if self._start_error is not None:
            raise self._start_error
        self.system_prompt = system_prompt
        self.history = history
        self._state = NovaSessionState.ACTIVE

    async def send_text(self, text: str) -> None:
        self.sent_text.append(text)

    async def start_audio_input(self) -> None:
        self.audio_starts += 1

    async def send_audio(self, pcm16le_16khz: bytes) -> None:
        if self._send_error is not None:
            raise self._send_error
        self.sent_audio.append(pcm16le_16khz)

    async def events(self) -> AsyncIterator[NovaServerEvent]:
        while True:
            event = await self._events.get()
            if event is None:
                return
            yield event

    async def finish_input(self) -> None:
        return None

    async def close(self) -> None:
        async with self._close_lock:
            if self._state is NovaSessionState.CLOSED:
                return
            self.close_started.set()
            await self._close_gate.wait()
            self.close_calls += 1
            self._state = NovaSessionState.CLOSED
            await self._events.put(None)
            if self._close_error is not None:
                raise self._close_error

    async def emit(self, event: NovaServerEvent) -> None:
        await self._events.put(event)

    def release_start(self) -> None:
        self._start_gate.set()

    def release_close(self) -> None:
        self._close_gate.set()


@pytest.mark.asyncio
async def test_rotation_replays_final_history_and_buffer_then_rejects_retired_output() -> None:
    transports: list[FakeTransport] = []

    def factory() -> FakeTransport:
        transport = FakeTransport()
        transports.append(transport)
        return transport

    manager = ContinuingNovaSonicTransport(
        factory=factory,
        rotation_seconds=0.02,
        audio_buffer_max_bytes=8,
    )
    await manager.start(system_prompt="Keep responses brief.")
    await manager.send_text("Hello")
    await manager.start_audio_input()
    await manager.send_audio(b"\x01\x00\x02\x00")
    await manager.send_audio(b"\x03\x00\x04\x00")
    await manager.send_audio(b"\x05\x00\x06\x00")

    caller = FinalTranscript(
        speaker=TranscriptSpeaker.CALLER,
        text="A caller turn",
        source_event_id="caller-1",
    )
    agent = FinalTranscript(
        speaker=TranscriptSpeaker.AGENT,
        text="An agent turn",
        source_event_id="agent-1",
    )
    await transports[0].emit(caller)
    await transports[0].emit(agent)
    assert await _next(manager) == caller
    assert await _next(manager) == agent

    started = await _next_of_type(manager, ContinuationStarted)
    succeeded = await _next_of_type(manager, ContinuationSucceeded)
    assert started.generation == 1
    assert succeeded.generation == 1
    assert succeeded.buffered_pcm16_bytes == 8
    assert len(transports) == 2
    assert transports[1].system_prompt == "Keep responses brief."
    assert transports[0].sent_text == ["Hello"]
    assert transports[1].sent_text == []
    assert transports[1].history == (
        ConversationHistoryTurn(
            role=ConversationRole.USER,
            text="A caller turn",
            source_event_id="caller-1",
        ),
        ConversationHistoryTurn(
            role=ConversationRole.ASSISTANT,
            text="An agent turn",
            source_event_id="agent-1",
        ),
    )
    assert transports[1].sent_audio == [b"\x03\x00\x04\x00\x05\x00\x06\x00"]
    assert transports[0].close_calls == 1

    await transports[0].emit(OutputAudio(pcm16le_24khz=b"\x09\x00"))
    current = OutputAudio(pcm16le_24khz=b"\x07\x00")
    await transports[1].emit(current)
    assert await _next(manager) == current

    await manager.send_audio(b"\x08\x00")
    assert transports[0].sent_audio == [
        b"\x01\x00\x02\x00",
        b"\x03\x00\x04\x00",
        b"\x05\x00\x06\x00",
    ]
    assert transports[1].sent_audio[-1] == b"\x08\x00"
    await manager.close()
    await manager.close()
    assert transports[1].close_calls == 1


@pytest.mark.asyncio
async def test_replacement_start_failure_is_bounded_and_keeps_old_stream_owned() -> None:
    initial = FakeTransport()
    replacement = FakeTransport(start_error=RuntimeError("synthetic"))
    transports = [initial, replacement]
    manager = ContinuingNovaSonicTransport(
        factory=lambda: transports.pop(0),
        rotation_seconds=0.01,
    )
    await manager.start(system_prompt="Test prompt")
    await manager.start_audio_input()

    assert isinstance(await _next_of_type(manager, ContinuationStarted), ContinuationStarted)
    failed = await _next_of_type(manager, ContinuationFailed)
    assert failed.phase == "STARTUP"
    assert manager.state is NovaSessionState.ACTIVE

    await manager.send_audio(b"\x01\x00")
    assert initial.sent_audio == [b"\x01\x00"]
    await manager.close()
    assert initial.close_calls == 1


@pytest.mark.asyncio
async def test_handoff_audio_failure_does_not_replace_the_old_stream() -> None:
    initial = FakeTransport()
    replacement = FakeTransport(send_error=RuntimeError("synthetic"))
    transports = [initial, replacement]
    manager = ContinuingNovaSonicTransport(
        factory=lambda: transports.pop(0),
        rotation_seconds=0.01,
    )
    await manager.start(system_prompt="Test prompt")
    await manager.start_audio_input()
    await manager.send_audio(b"\x01\x00")

    assert isinstance(await _next_of_type(manager, ContinuationStarted), ContinuationStarted)
    failed = await _next_of_type(manager, ContinuationFailed)
    assert failed.phase == "HANDOFF"
    assert manager.state is NovaSessionState.ACTIVE
    assert replacement.close_calls == 1

    await manager.close()
    assert initial.close_calls == 1


@pytest.mark.asyncio
async def test_close_cancels_blocked_replacement_start_and_closes_both_generations() -> None:
    initial = FakeTransport()
    replacement = FakeTransport(block_start=True)
    transports = [initial, replacement]
    manager = ContinuingNovaSonicTransport(
        factory=lambda: transports.pop(0),
        rotation_seconds=0.01,
    )
    await manager.start(system_prompt="Test prompt")

    assert isinstance(await _next_of_type(manager, ContinuationStarted), ContinuationStarted)
    await asyncio.wait_for(replacement.start_started.wait(), timeout=1)
    await asyncio.wait_for(manager.close(), timeout=1)

    assert initial.close_calls == 1
    assert replacement.close_calls == 1
    assert manager.state is NovaSessionState.CLOSED


@pytest.mark.asyncio
async def test_close_during_retirement_closes_old_and_new_once() -> None:
    initial = FakeTransport(block_close=True)
    replacement = FakeTransport()
    transports = [initial, replacement]
    manager = ContinuingNovaSonicTransport(
        factory=lambda: transports.pop(0),
        rotation_seconds=0.01,
    )
    await manager.start(system_prompt="Test prompt")
    await manager.start_audio_input()

    assert isinstance(await _next_of_type(manager, ContinuationStarted), ContinuationStarted)
    await asyncio.wait_for(initial.close_started.wait(), timeout=1)
    close_task = asyncio.create_task(manager.close())
    await asyncio.wait_for(replacement.close_started.wait(), timeout=1)
    initial.release_close()
    await asyncio.wait_for(close_task, timeout=1)

    assert initial.close_calls == 1
    assert replacement.close_calls == 1
    assert manager.state is NovaSessionState.CLOSED


@pytest.mark.asyncio
async def test_history_limit_evicts_oldest_whole_final_turn() -> None:
    transports: list[FakeTransport] = []

    def factory() -> FakeTransport:
        transport = FakeTransport()
        transports.append(transport)
        return transport

    manager = ContinuingNovaSonicTransport(
        factory=factory,
        rotation_seconds=0.01,
        history_max_bytes=20,
    )
    await manager.start(system_prompt="Test prompt")
    first = FinalTranscript(
        speaker=TranscriptSpeaker.CALLER,
        text="first-turn",
        source_event_id="first",
    )
    second = FinalTranscript(
        speaker=TranscriptSpeaker.AGENT,
        text="second",
        source_event_id="second",
    )
    await transports[0].emit(first)
    await transports[0].emit(second)
    assert await _next(manager) == first
    assert await _next(manager) == second
    await _next_of_type(manager, ContinuationStarted)
    await _next_of_type(manager, ContinuationSucceeded)

    assert transports[1].history == (
        ConversationHistoryTurn(
            role=ConversationRole.ASSISTANT,
            text="second",
            source_event_id="second",
        ),
    )
    await manager.close()


def test_history_event_shape_is_noninteractive_ordered_and_role_preserving() -> None:
    history = (
        ConversationHistoryTurn(
            role=ConversationRole.USER,
            text="Caller context",
            source_event_id="caller-context",
        ),
        ConversationHistoryTurn(
            role=ConversationRole.ASSISTANT,
            text="Agent context",
            source_event_id="agent-context",
        ),
    )

    events = [
        json.loads(payload)
        for payload in build_history_events(
            prompt_name="prompt-1",
            history=history,
        )
    ]

    assert len(events) == 6
    for index, turn in enumerate(history):
        content_start = events[index * 3]["event"]["contentStart"]
        text_input = events[index * 3 + 1]["event"]["textInput"]
        content_end = events[index * 3 + 2]["event"]["contentEnd"]
        assert content_start["interactive"] is False
        assert content_start["role"] == turn.role.value
        assert text_input["content"] == turn.text
        assert content_start["contentName"] == text_input["contentName"]
        assert content_start["contentName"] == content_end["contentName"]


async def _next(manager: ContinuingNovaSonicTransport) -> NovaServerEvent:
    return await asyncio.wait_for(anext(manager.events()), timeout=1)


async def _next_of_type[
    EventT: NovaServerEvent,
](
    manager: ContinuingNovaSonicTransport,
    event_type: type[EventT],
) -> EventT:
    while True:
        event = await _next(manager)
        if isinstance(event, event_type):
            return event
