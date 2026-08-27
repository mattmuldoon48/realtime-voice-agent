"""Deterministic simultaneous-call ownership and failure-isolation proof."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import cast

import pytest
import structlog
from structlog.typing import FilteringBoundLogger

from realtime_voice_agent.audio.codecs import encode_twilio_mulaw_payload
from realtime_voice_agent.nova.events import (
    CompletionEnded,
    NovaServerEvent,
    NovaSessionState,
    OutputAudio,
)
from realtime_voice_agent.observability.models import MetricDatum, MetricName
from realtime_voice_agent.persistence.models import (
    PersistedSessionStatus,
    PersonaSnapshot,
    SessionStart,
    SessionTerminal,
    TranscriptTurn,
    TranscriptView,
)
from realtime_voice_agent.telephony.events import (
    MarkEvent,
    MediaEvent,
    MediaFormat,
    StartEvent,
    StopEvent,
    TwilioMarkCommand,
    TwilioMediaCommand,
)
from realtime_voice_agent.telephony.session import CallOutcome, CallSession, CallSessionError
from realtime_voice_agent.transcript import FinalTranscript, TranscriptSpeaker


class MemoryTelemetry:
    def __init__(self) -> None:
        self.metrics: list[MetricDatum] = []

    def publish_metric(self, metric: MetricDatum) -> bool:
        self.metrics.append(metric)
        return True

    def publish_log(self, event: dict[str, object]) -> bool:
        del event
        return True


class MemoryRepository:
    def __init__(self) -> None:
        self.sessions: list[SessionStart] = []
        self.activations: list[tuple[str, str]] = []
        self.turns: list[TranscriptTurn] = []
        self.terminals: list[SessionTerminal] = []

    def create_session(self, session: SessionStart) -> None:
        self.sessions.append(session)

    def mark_session_active(self, session_id: str, activated_at: str) -> None:
        self.activations.append((session_id, activated_at))

    def append_transcript_turn(self, turn: TranscriptTurn) -> bool:
        self.turns.append(turn)
        return True

    def finish_session(self, terminal: SessionTerminal) -> bool:
        self.terminals.append(terminal)
        return True

    def get_transcript(
        self,
        *,
        session_id: str | None = None,
        call_sid: str | None = None,
    ) -> TranscriptView:
        del session_id, call_sid
        raise NotImplementedError


class ConcurrentNovaTransport:
    def __init__(self, *, fail_audio_write: bool = False) -> None:
        self._state = NovaSessionState.NEW
        self._fail_audio_write = fail_audio_write
        self._events: asyncio.Queue[NovaServerEvent | None] = asyncio.Queue()
        self.started = asyncio.Event()
        self.audio_received = asyncio.Event()
        self.sent_audio: list[bytes] = []
        self.close_calls = 0

    @property
    def state(self) -> NovaSessionState:
        return self._state

    async def start(
        self,
        *,
        system_prompt: str,
        history: tuple[object, ...] = (),
    ) -> None:
        assert system_prompt
        self._state = NovaSessionState.ACTIVE
        self.started.set()

    async def start_audio_input(self) -> None:
        assert self._state is NovaSessionState.ACTIVE

    async def send_audio(self, pcm16le_16khz: bytes) -> None:
        if self._fail_audio_write:
            raise RuntimeError("synthetic Nova write failure")
        self.sent_audio.append(pcm16le_16khz)
        self.audio_received.set()

    async def events(self) -> AsyncIterator[NovaServerEvent]:
        while True:
            event = await self._events.get()
            if event is None:
                return
            yield event

    async def finish_input(self) -> None:
        return None

    async def close(self) -> None:
        self.close_calls += 1
        self._state = NovaSessionState.CLOSED
        await self._events.put(None)

    async def emit(self, event: NovaServerEvent) -> None:
        await self._events.put(event)


@dataclass(slots=True)
class ConcurrentCase:
    index: int
    mode: str
    session: CallSession
    nova: ConcurrentNovaTransport
    repository: MemoryRepository
    telemetry: MemoryTelemetry
    outbound_payload: str | None = None


_PERSONA = PersonaSnapshot(
    persona_id="synthetic-concurrency",
    name="Synthetic concurrency",
    system_prompt="Return the synthetic test response.",
    voice_id="matthew",
    version=1,
)


@pytest.mark.asyncio
async def test_ten_simultaneous_sessions_isolate_state_audio_transcripts_and_failures() -> None:
    cases = [_case(index) for index in range(10)]

    await asyncio.gather(*(_drive(case) for case in cases))

    healthy = cases[2:]
    assert all(case.session.snapshot.outcome is CallOutcome.SUCCEEDED for case in healthy)
    assert all(
        case.repository.terminals[0].status is PersistedSessionStatus.COMPLETED for case in healthy
    )
    assert [case.session.failure_code for case in cases[:2]] == [
        "NOVA_STREAM_FAILED",
        "TWILIO_OUTBOUND_QUEUE_OVERFLOW",
    ]
    assert all(
        case.repository.terminals[0].status is PersistedSessionStatus.FAILED for case in cases[:2]
    )

    assert len({case.session.snapshot.session_id for case in cases}) == 10
    assert len({case.session.snapshot.call_sid for case in cases}) == 10
    assert len({case.session.snapshot.stream_sid for case in cases}) == 10
    assert len({case.outbound_payload for case in healthy}) == 8
    assert len({case.nova.sent_audio[0] for case in healthy}) == 8
    assert all(case.nova.close_calls == 1 for case in cases)

    for case in healthy:
        assert [turn.turn_number for turn in case.repository.turns] == [1, 2]
        assert [turn.text for turn in case.repository.turns] == [
            f"synthetic caller {case.index}",
            f"synthetic agent {case.index}",
        ]
        names = [metric.name for metric in case.telemetry.metrics]
        assert names.count(MetricName.CALLS_COMPLETED) == 1
        assert MetricName.CALLS_FAILED not in names


def _case(index: int) -> ConcurrentCase:
    mode = "nova_failure" if index == 0 else "overflow" if index == 1 else "healthy"
    nova = ConcurrentNovaTransport(fail_audio_write=mode == "nova_failure")
    repository = MemoryRepository()
    telemetry = MemoryTelemetry()
    logger = cast(FilteringBoundLogger, structlog.get_logger())
    session = CallSession(
        logger=logger,
        expected_twilio_account_sid="synthetic-account",
        malformed_frame_limit=1,
        audio_queue_max_frames=4,
        outbound_queue_max_frames=1 if mode == "overflow" else 4,
        nova=nova,
        persona=_PERSONA,
        model_id="amazon.nova-2-sonic-v1:0",
        session_repository=repository,
        persistence_queue_max_events=8,
        transcript_retention_days=7,
        telemetry=telemetry,
        environment="test",
        clock=lambda: datetime.now(UTC),
        session_id_factory=lambda: f"synthetic-session-{index}",
        cleanup_timeout_seconds=1.0,
    )
    return ConcurrentCase(
        index=index,
        mode=mode,
        session=session,
        nova=nova,
        repository=repository,
        telemetry=telemetry,
    )


async def _drive(case: ConcurrentCase) -> None:
    case.session.handle_event(_start(case.index))
    await asyncio.wait_for(case.nova.started.wait(), timeout=1)

    if case.mode == "overflow":
        await case.nova.emit(OutputAudio(pcm16le_24khz=b"\x01\x00" * 480))
        await case.nova.emit(OutputAudio(pcm16le_24khz=b"\x02\x00" * 480))
        await _expect_failure(case.session)
        await case.session.close()
        return

    case.session.handle_event(_media(case.index))
    if case.mode == "nova_failure":
        await _expect_failure(case.session)
        await case.session.close()
        return

    await asyncio.wait_for(case.nova.audio_received.wait(), timeout=1)
    output_sample = ((case.index - 5) * 4_000).to_bytes(2, byteorder="little", signed=True)
    await case.nova.emit(OutputAudio(pcm16le_24khz=output_sample * 480))
    await case.nova.emit(
        FinalTranscript(
            speaker=TranscriptSpeaker.CALLER,
            text=f"synthetic caller {case.index}",
            source_event_id=f"caller-{case.index}",
        )
    )
    await case.nova.emit(
        FinalTranscript(
            speaker=TranscriptSpeaker.AGENT,
            text=f"synthetic agent {case.index}",
            source_event_id=f"agent-{case.index}",
        )
    )
    await case.nova.emit(CompletionEnded())

    media = await asyncio.wait_for(case.session.next_outbound_message(), timeout=1)
    mark = await asyncio.wait_for(case.session.next_outbound_message(), timeout=1)
    assert isinstance(media, TwilioMediaCommand)
    assert isinstance(mark, TwilioMarkCommand)
    case.outbound_payload = media.payload
    case.session.record_outbound_sent(media)
    case.session.record_outbound_sent(mark)
    case.session.handle_event(MarkEvent(sequence_number=3, name=mark.name))
    case.session.handle_event(StopEvent(sequence_number=4))
    await case.session.close()


async def _expect_failure(session: CallSession) -> None:
    with pytest.raises(CallSessionError):
        await asyncio.wait_for(session.wait_for_failure(), timeout=1)


def _start(index: int) -> StartEvent:
    return StartEvent(
        sequence_number=1,
        stream_sid=f"synthetic-stream-{index}",
        call_sid=f"synthetic-call-{index}",
        account_sid="synthetic-account",
        media_format=MediaFormat(
            encoding="audio/x-mulaw",
            sample_rate=8_000,
            channels=1,
        ),
    )


def _media(index: int) -> MediaEvent:
    sample = ((index - 5) * 4_000).to_bytes(2, byteorder="little", signed=True)
    return MediaEvent(
        sequence_number=2,
        chunk=1,
        timestamp_ms=20,
        payload=encode_twilio_mulaw_payload(sample * 160),
        track="inbound",
    )
