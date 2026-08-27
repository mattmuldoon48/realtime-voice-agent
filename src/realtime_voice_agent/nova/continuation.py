"""Bounded make-before-break continuation for one Nova voice call."""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass

from realtime_voice_agent.nova.events import (
    ContinuationFailed,
    ContinuationStarted,
    ContinuationSucceeded,
    NovaServerEvent,
    NovaSessionState,
)
from realtime_voice_agent.nova.transport import NovaSonicTransport
from realtime_voice_agent.transcript import ConversationHistoryTurn, FinalTranscript

NovaTransportFactory = Callable[[], NovaSonicTransport]

_HISTORY_SINGLE_TURN_MAX_BYTES = 51_200
_HISTORY_TOTAL_MAX_BYTES = 204_800
_AUDIO_BUFFER_MAX_BYTES = 96_000
_EVENT_QUEUE_MAX_ITEMS = 256


@dataclass(frozen=True, slots=True)
class _ReaderFailure:
    generation: int
    error: Exception


type _QueuedEvent = NovaServerEvent | _ReaderFailure | None


class ContinuingNovaSonicTransport:
    """Rotate Nova streams without changing the owning Twilio call session."""

    def __init__(
        self,
        *,
        factory: NovaTransportFactory,
        rotation_seconds: float,
        history_max_bytes: int = _HISTORY_TOTAL_MAX_BYTES,
        audio_buffer_max_bytes: int = _AUDIO_BUFFER_MAX_BYTES,
    ) -> None:
        if rotation_seconds <= 0:
            raise ValueError("rotation_seconds must be positive")
        if rotation_seconds > 450:
            raise ValueError("rotation_seconds must be at most 450")
        if history_max_bytes <= 0:
            raise ValueError("history_max_bytes must be positive")
        if audio_buffer_max_bytes <= 0 or audio_buffer_max_bytes % 2:
            raise ValueError("audio_buffer_max_bytes must be positive and sample-aligned")

        self._factory = factory
        self._rotation_seconds = rotation_seconds
        self._history_max_bytes = history_max_bytes
        self._audio_buffer_max_bytes = audio_buffer_max_bytes
        self._active = factory()
        self._pending: NovaSonicTransport | None = None
        self._retiring: NovaSonicTransport | None = None
        self._system_prompt: str | None = None
        self._generation = 0
        self._audio_started = False
        self._closed = False
        self._rotation_failed = False
        self._suppressed_generation: int | None = None
        self._history: deque[ConversationHistoryTurn] = deque()
        self._history_bytes = 0
        self._history_source_ids: set[str] = set()
        self._rolling_audio: deque[bytes] = deque()
        self._rolling_audio_bytes = 0
        self._events: asyncio.Queue[_QueuedEvent] = asyncio.Queue(maxsize=_EVENT_QUEUE_MAX_ITEMS)
        self._route_lock = asyncio.Lock()
        self._rotation_lock = asyncio.Lock()
        self._close_lock = asyncio.Lock()
        self._reader_tasks: set[asyncio.Task[None]] = set()
        self._rotation_task: asyncio.Task[None] | None = None

    @property
    def state(self) -> NovaSessionState:
        """Return the active underlying stream state."""
        if self._closed:
            return NovaSessionState.CLOSED
        return self._active.state

    async def start(
        self,
        *,
        system_prompt: str,
        history: tuple[ConversationHistoryTurn, ...] = (),
    ) -> None:
        """Start the initial generation and its bounded rotation timer."""
        if self._system_prompt is not None:
            raise RuntimeError("Nova continuation transport has already started")
        self._system_prompt = system_prompt
        for turn in history:
            self._append_history(turn)
        await self._active.start(system_prompt=system_prompt, history=tuple(self._history))
        self._start_reader(self._active, generation=0)
        self._rotation_task = asyncio.create_task(
            self._rotation_loop(),
            name="nova-continuation-timer",
        )

    async def send_text(self, text: str) -> None:
        """Route one initial interactive text turn to the first generation."""
        async with self._route_lock:
            await self._active.send_text(text)

    async def start_audio_input(self) -> None:
        """Open audio input on the initial active stream."""
        async with self._route_lock:
            await self._active.start_audio_input()
            self._audio_started = True

    async def send_audio(self, pcm16le_16khz: bytes) -> None:
        """Route one input chunk and retain only the most recent three seconds."""
        async with self._route_lock:
            self._append_rolling_audio(pcm16le_16khz)
            await self._active.send_audio(pcm16le_16khz)

    async def events(self) -> AsyncIterator[NovaServerEvent]:
        """Yield only events owned by the current generation."""
        while True:
            item = await self._events.get()
            if item is None:
                return
            if isinstance(item, _ReaderFailure):
                if item.generation == self._generation and not self._closed:
                    raise item.error
                continue
            yield item

    async def finish_input(self) -> None:
        """Finish only the currently active input content block."""
        async with self._route_lock:
            await self._active.finish_input()

    async def close(self) -> None:
        """Idempotently stop rotation, readers, and every live generation."""
        async with self._close_lock:
            if self._closed:
                return
            self._closed = True
            rotation_task = self._rotation_task
            if rotation_task is not None and rotation_task is not asyncio.current_task():
                rotation_task.cancel()

            async with self._route_lock:
                transports = {id(self._active): self._active}
                if self._pending is not None:
                    transports[id(self._pending)] = self._pending
                if self._retiring is not None:
                    transports[id(self._retiring)] = self._retiring
                self._pending = None
                self._retiring = None

            results = await asyncio.gather(
                *(transport.close() for transport in transports.values()),
                return_exceptions=True,
            )
            if rotation_task is not None and rotation_task is not asyncio.current_task():
                await asyncio.gather(rotation_task, return_exceptions=True)
            reader_tasks = tuple(self._reader_tasks)
            for task in reader_tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*reader_tasks, return_exceptions=True)
            self._reader_tasks.difference_update(reader_tasks)
            self._put_terminal_event()

            error = next((result for result in results if isinstance(result, Exception)), None)
            if error is not None:
                raise error

    async def _rotation_loop(self) -> None:
        try:
            while not self._closed and not self._rotation_failed:
                await asyncio.sleep(self._rotation_seconds)
                await self._rotate()
        except asyncio.CancelledError:
            raise

    async def _rotate(self) -> None:
        async with self._rotation_lock:
            if self._closed or self._rotation_failed:
                return
            system_prompt = self._system_prompt
            if system_prompt is None:
                raise RuntimeError("Nova continuation transport has not started")

            next_generation = self._generation + 1
            old_generation = self._generation
            history = tuple(self._history)
            await self._events.put(
                ContinuationStarted(
                    generation=next_generation,
                    history_turns=len(history),
                )
            )
            self._suppressed_generation = old_generation
            try:
                replacement = self._factory()
            except Exception:
                self._rotation_failed = True
                self._suppressed_generation = None
                await self._events.put(
                    ContinuationFailed(
                        generation=next_generation,
                        phase="STARTUP",
                    )
                )
                return
            self._pending = replacement
            try:
                await replacement.start(system_prompt=system_prompt, history=history)
                if self._audio_started:
                    await replacement.start_audio_input()
            except Exception:
                self._rotation_failed = True
                self._suppressed_generation = None
                self._pending = None
                await asyncio.gather(replacement.close(), return_exceptions=True)
                await self._events.put(
                    ContinuationFailed(
                        generation=next_generation,
                        phase="STARTUP",
                    )
                )
                return

            handoff_failed = False
            old_transport: NovaSonicTransport
            buffered_audio = b""
            try:
                async with self._route_lock:
                    old_transport = self._active
                    buffered_audio = b"".join(self._rolling_audio)
                    if buffered_audio:
                        await replacement.send_audio(buffered_audio)
                    self._retiring = old_transport
                    self._active = replacement
                    self._pending = None
                    self._generation = next_generation
                    self._start_reader(replacement, generation=next_generation)
                    self._suppressed_generation = None
            except Exception:
                handoff_failed = True
                self._rotation_failed = True
                self._suppressed_generation = None
                self._pending = None

            if handoff_failed:
                await asyncio.gather(replacement.close(), return_exceptions=True)
                await self._events.put(
                    ContinuationFailed(
                        generation=next_generation,
                        phase="HANDOFF",
                    )
                )
                return

            retired_results = await asyncio.gather(
                old_transport.close(),
                return_exceptions=True,
            )
            self._retiring = None
            retired_cleanup_failed = isinstance(retired_results[0], Exception)
            await self._events.put(
                ContinuationSucceeded(
                    generation=next_generation,
                    history_turns=len(history),
                    buffered_pcm16_bytes=len(buffered_audio),
                    retired_cleanup_failed=retired_cleanup_failed,
                )
            )

    def _start_reader(self, transport: NovaSonicTransport, *, generation: int) -> None:
        task = asyncio.create_task(
            self._read_generation(transport, generation=generation),
            name=f"nova-generation-{generation}-reader",
        )
        self._reader_tasks.add(task)
        task.add_done_callback(self._reader_tasks.discard)

    async def _read_generation(
        self,
        transport: NovaSonicTransport,
        *,
        generation: int,
    ) -> None:
        try:
            async for event in transport.events():
                if self._closed or generation != self._generation:
                    continue
                if generation == self._suppressed_generation:
                    continue
                if isinstance(event, FinalTranscript):
                    self._append_history(ConversationHistoryTurn.from_final_transcript(event))
                await self._events.put(event)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            if not self._closed and generation != self._suppressed_generation:
                await self._events.put(_ReaderFailure(generation=generation, error=error))
        else:
            if (
                not self._closed
                and generation == self._generation
                and generation != self._suppressed_generation
            ):
                await self._events.put(
                    _ReaderFailure(
                        generation=generation,
                        error=RuntimeError("Active Nova generation ended unexpectedly"),
                    )
                )

    def _append_history(self, turn: ConversationHistoryTurn) -> None:
        if turn.source_event_id in self._history_source_ids:
            return
        bounded = _bounded_turn(turn)
        size = _history_turn_size(bounded)
        if size > self._history_max_bytes:
            return
        self._history.append(bounded)
        self._history_source_ids.add(bounded.source_event_id)
        self._history_bytes += size
        while self._history and self._history_bytes > self._history_max_bytes:
            removed = self._history.popleft()
            self._history_source_ids.discard(removed.source_event_id)
            self._history_bytes -= _history_turn_size(removed)

    def _append_rolling_audio(self, audio: bytes) -> None:
        if not audio:
            return
        if len(audio) > self._audio_buffer_max_bytes:
            audio = audio[-self._audio_buffer_max_bytes :]
            if len(audio) % 2:
                audio = audio[1:]
        self._rolling_audio.append(audio)
        self._rolling_audio_bytes += len(audio)
        while self._rolling_audio and self._rolling_audio_bytes > self._audio_buffer_max_bytes:
            removed = self._rolling_audio.popleft()
            self._rolling_audio_bytes -= len(removed)

    def _put_terminal_event(self) -> None:
        try:
            self._events.put_nowait(None)
        except asyncio.QueueFull:
            try:
                self._events.get_nowait()
            except asyncio.QueueEmpty:
                pass
            self._events.put_nowait(None)


def _bounded_turn(turn: ConversationHistoryTurn) -> ConversationHistoryTurn:
    content = turn.text.encode("utf-8")
    if len(content) <= _HISTORY_SINGLE_TURN_MAX_BYTES:
        return turn
    marker = "... [truncated]"
    marker_bytes = marker.encode("utf-8")
    text = content[: _HISTORY_SINGLE_TURN_MAX_BYTES - len(marker_bytes)].decode(
        "utf-8",
        errors="ignore",
    )
    return ConversationHistoryTurn(
        role=turn.role,
        text=f"{text}{marker}",
        source_event_id=turn.source_event_id,
    )


def _history_turn_size(turn: ConversationHistoryTurn) -> int:
    return len(turn.text.encode("utf-8")) + len(turn.role.value.encode("ascii"))
