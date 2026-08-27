"""Unit tests for the bounded caller-audio-to-Nova call session."""

from __future__ import annotations

import asyncio
import base64
import threading
from collections.abc import AsyncIterator, Callable
from datetime import UTC, datetime, timedelta
from typing import cast

import pytest
import structlog
from structlog.typing import FilteringBoundLogger

from realtime_voice_agent.audio.codecs import encode_twilio_mulaw_payload
from realtime_voice_agent.nova.events import (
    CompletionEnded,
    ContinuationFailed,
    ContinuationStarted,
    ContinuationSucceeded,
    InterruptionStarted,
    NovaServerEvent,
    NovaSessionState,
    OutputAudio,
)
from realtime_voice_agent.observability.models import MetricDatum, MetricName, TelemetryPublisher
from realtime_voice_agent.persistence.errors import PersistenceError
from realtime_voice_agent.persistence.models import (
    PersistedSessionStatus,
    PersonaSnapshot,
    SessionStart,
    SessionTerminal,
    TranscriptTurn,
    TranscriptView,
)
from realtime_voice_agent.telephony.events import (
    DtmfEvent,
    MarkEvent,
    MediaEvent,
    MediaFormat,
    StartEvent,
    StopEvent,
    TwilioClearCommand,
    TwilioMarkCommand,
    TwilioMediaCommand,
)
from realtime_voice_agent.telephony.session import (
    CallOutcome,
    CallSession,
    CallSessionError,
    CallSessionState,
    CallTerminationReason,
)
from realtime_voice_agent.transcript import FinalTranscript, TranscriptSpeaker


class FakeTelemetry:
    def __init__(self) -> None:
        self.metrics: list[MetricDatum] = []

    def publish_metric(self, metric: MetricDatum) -> bool:
        self.metrics.append(metric)
        return True

    def publish_log(self, event: dict[str, object]) -> bool:
        del event
        return True


class FakeNovaTransport:
    def __init__(
        self,
        *,
        block_start: bool = False,
        start_error: Exception | None = None,
        block_close: bool = False,
        send_error: Exception | None = None,
        events_on_close: list[NovaServerEvent] | None = None,
        end_events_on_close: bool = True,
        close_cancel_delay_seconds: float = 0.0,
        event_cancel_delay_seconds: float = 0.0,
    ) -> None:
        self._state = NovaSessionState.NEW
        self._start_gate = asyncio.Event()
        if not block_start:
            self._start_gate.set()
        self._start_error = start_error
        self._send_error = send_error
        self._close_gate = asyncio.Event()
        self._events_on_close = events_on_close or []
        self._end_events_on_close = end_events_on_close
        self._close_cancel_delay_seconds = close_cancel_delay_seconds
        self._event_cancel_delay_seconds = event_cancel_delay_seconds
        if not block_close:
            self._close_gate.set()
        self.started = asyncio.Event()
        self.close_started = asyncio.Event()
        self.audio_received = asyncio.Event()
        self.close_finished = asyncio.Event()
        self.events_cancelled = asyncio.Event()
        self.sent_audio: list[bytes] = []
        self.sent_text: list[str] = []
        self.output_events: asyncio.Queue[NovaServerEvent | None] = asyncio.Queue()
        self.close_calls = 0
        self.system_prompts: list[str] = []

    @property
    def state(self) -> NovaSessionState:
        return self._state

    async def start(
        self,
        *,
        system_prompt: str,
        history: tuple[object, ...] = (),
    ) -> None:
        self.system_prompts.append(system_prompt)
        assert system_prompt
        if self._start_error is not None:
            raise self._start_error
        await self._start_gate.wait()
        self.started.set()
        self._state = NovaSessionState.ACTIVE

    async def send_text(self, text: str) -> None:
        self.sent_text.append(text)

    async def start_audio_input(self) -> None:
        assert self._state is NovaSessionState.ACTIVE

    async def send_audio(self, pcm16le_16khz: bytes) -> None:
        if self._send_error is not None:
            raise self._send_error
        self.sent_audio.append(pcm16le_16khz)
        self.audio_received.set()

    async def events(self) -> AsyncIterator[NovaServerEvent]:
        try:
            while True:
                event = await self.output_events.get()
                if event is None:
                    return
                yield event
        except asyncio.CancelledError:
            if self._event_cancel_delay_seconds > 0:
                await asyncio.sleep(self._event_cancel_delay_seconds)
                return
            raise
        finally:
            self.events_cancelled.set()

    async def finish_input(self) -> None:
        return None

    async def close(self) -> None:
        self.close_calls += 1
        self.close_started.set()
        try:
            await self._close_gate.wait()
        except asyncio.CancelledError:
            if self._close_cancel_delay_seconds <= 0:
                raise
            await asyncio.sleep(self._close_cancel_delay_seconds)
        finally:
            self._state = NovaSessionState.CLOSED
            self.close_finished.set()
        for event in self._events_on_close:
            await self.output_events.put(event)
        if self._end_events_on_close:
            await self.output_events.put(None)

    def allow_close(self) -> None:
        self._close_gate.set()


class FakeSessionRepository:
    def __init__(
        self,
        *,
        create_error: Exception | None = None,
        create_errors: list[Exception] | None = None,
    ) -> None:
        self.create_error = create_error
        self.create_errors = create_errors or []
        self.create_calls = 0
        self.sessions: list[SessionStart] = []
        self.activations: list[tuple[str, str]] = []
        self.turns: list[TranscriptTurn] = []
        self.terminals: list[SessionTerminal] = []
        self.session_created = threading.Event()
        self.two_turns_written = threading.Event()

    def create_session(self, session: SessionStart) -> None:
        self.create_calls += 1
        if self.create_errors:
            raise self.create_errors.pop(0)
        if self.create_error is not None:
            raise self.create_error
        self.sessions.append(session)
        self.session_created.set()

    def mark_session_active(self, session_id: str, activated_at: str) -> None:
        self.activations.append((session_id, activated_at))

    def append_transcript_turn(self, turn: TranscriptTurn) -> bool:
        if any(
            existing.session_id == turn.session_id and existing.turn_number == turn.turn_number
            for existing in self.turns
        ):
            return False
        self.turns.append(turn)
        if len(self.turns) >= 2:
            self.two_turns_written.set()
        return True

    def finish_session(self, terminal: SessionTerminal) -> bool:
        if self.terminals:
            return False
        self.terminals.append(terminal)
        return True

    def get_transcript(
        self,
        *,
        session_id: str | None = None,
        call_sid: str | None = None,
    ) -> TranscriptView:
        raise NotImplementedError


_PERSONA = PersonaSnapshot(
    persona_id="concierge",
    name="Concierge",
    system_prompt="Answer as the configured concierge.",
    voice_id="matthew",
    version=3,
)


class StepClock:
    def __init__(self) -> None:
        self.current = datetime(2026, 8, 4, 12, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        timestamp = self.current
        self.current += timedelta(seconds=1)
        return timestamp


def _now() -> datetime:
    return datetime.now(UTC)


async def test_call_session_records_completed_lifecycle_and_timestamps() -> None:
    clock = StepClock()
    nova = FakeNovaTransport()
    session = _session(nova=nova, queue_max_frames=4, clock=clock)

    initial = session.snapshot
    assert initial.state is CallSessionState.STARTING
    assert initial.started_at == datetime(2026, 8, 4, 12, 0, tzinfo=UTC)
    assert initial.activated_at is None
    assert initial.ended_at is None

    session.handle_event(_start_event())
    active = session.snapshot
    assert active.state is CallSessionState.ACTIVE
    assert active.activated_at == datetime(2026, 8, 4, 12, 0, 1, tzinfo=UTC)

    session.handle_event(StopEvent(sequence_number=2))
    completed = session.snapshot
    assert completed.state is CallSessionState.COMPLETED
    assert completed.outcome is CallOutcome.SUCCEEDED
    assert completed.termination_reason is CallTerminationReason.TWILIO_STOP
    assert completed.ended_at == datetime(2026, 8, 4, 12, 0, 2, tzinfo=UTC)
    assert completed.failure_code is None

    await session.close()
    assert session.snapshot == completed


async def test_inbound_dtmf_is_ignored_without_terminating_the_call() -> None:
    nova = FakeNovaTransport()
    session = _session(nova=nova, queue_max_frames=4)
    session.handle_event(_start_event())

    session.handle_event(DtmfEvent(sequence_number=2, track="inbound_track"))

    assert session.state is CallSessionState.ACTIVE
    assert session.failure_code is None
    session.handle_event(StopEvent(sequence_number=3))
    await session.close()


async def test_initial_text_prompt_is_sent_once_before_caller_audio() -> None:
    nova = FakeNovaTransport()
    session = _session(
        nova=nova,
        queue_max_frames=4,
        initial_text_prompt="Hello",
    )

    session.handle_event(_start_event())
    await asyncio.wait_for(nova.started.wait(), timeout=1)
    await asyncio.sleep(0)

    assert nova.sent_text == ["Hello"]
    assert nova.sent_audio == []
    session.handle_event(StopEvent(sequence_number=2))
    await session.close()


async def test_start_does_not_enqueue_probe_audio() -> None:
    nova = FakeNovaTransport()
    session = _session(nova=nova, queue_max_frames=4)
    session.handle_event(_start_event())
    await asyncio.wait_for(nova.started.wait(), timeout=1)

    with pytest.raises(TimeoutError):
        await asyncio.wait_for(session.next_outbound_message(), timeout=0.01)

    await session.close()


async def test_nova_start_failure_is_normalized_and_cleanup_is_idempotent() -> None:
    nova = FakeNovaTransport(start_error=RuntimeError("synthetic startup failure"))
    session = _session(nova=nova, queue_max_frames=4)
    session.handle_event(_start_event())

    await _wait_for_failure(session)
    failed = session.snapshot
    assert failed.state is CallSessionState.FAILED
    assert failed.outcome is CallOutcome.FAILED
    assert failed.termination_reason is CallTerminationReason.NOVA_ERROR
    assert failed.failure_code == "NOVA_STREAM_FAILED"

    await session.close()
    await session.close()
    assert session.snapshot == failed
    assert nova.close_calls == 1


async def test_nova_mid_call_failure_cancels_sibling_worker() -> None:
    nova = FakeNovaTransport(send_error=RuntimeError("synthetic send failure"))
    session = _session(nova=nova, queue_max_frames=4)
    session.handle_event(_start_event())
    session.handle_event(_media_event(sequence_number=2, chunk=1))

    await _wait_for_failure(session)
    await asyncio.wait_for(nova.events_cancelled.wait(), timeout=1)

    assert session.state is CallSessionState.FAILED
    assert session.snapshot.termination_reason is CallTerminationReason.NOVA_ERROR
    assert session.failure_code == "NOVA_STREAM_FAILED"
    await session.close()


async def test_first_terminal_transition_wins_disconnect_failure_race() -> None:
    nova = FakeNovaTransport()
    session = _session(nova=nova, queue_max_frames=4)
    session.handle_event(_start_event())

    session.disconnect()
    disconnected = session.snapshot
    session.fail("CALL_BRIDGE_FAILED", CallTerminationReason.INTERNAL_ERROR)
    await session.close()

    assert session.snapshot == disconnected
    assert disconnected.state is CallSessionState.DISCONNECTED
    assert disconnected.outcome is CallOutcome.DISCONNECTED
    assert disconnected.termination_reason is CallTerminationReason.WEBSOCKET_DISCONNECT
    assert disconnected.failure_code is None


async def test_two_call_sessions_keep_identifiers_and_output_isolated() -> None:
    first_nova = FakeNovaTransport()
    second_nova = FakeNovaTransport()
    first = _session(nova=first_nova, queue_max_frames=4)
    second = _session(nova=second_nova, queue_max_frames=4)
    first.handle_event(_start_event(stream_digit="1"))
    second.handle_event(_start_event(stream_digit="2"))
    await asyncio.wait_for(first_nova.started.wait(), timeout=1)
    await asyncio.wait_for(second_nova.started.wait(), timeout=1)

    await first_nova.output_events.put(OutputAudio(pcm16le_24khz=b"\x01\x00" * 480))
    await second_nova.output_events.put(OutputAudio(pcm16le_24khz=b"\x02\x00" * 480))
    first_audio = await asyncio.wait_for(first.next_outbound_message(), timeout=1)
    second_audio = await asyncio.wait_for(second.next_outbound_message(), timeout=1)

    assert isinstance(first_audio, TwilioMediaCommand)
    assert isinstance(second_audio, TwilioMediaCommand)
    assert first_audio.stream_sid == first.snapshot.stream_sid
    assert second_audio.stream_sid == second.snapshot.stream_sid
    assert first_audio.stream_sid != second_audio.stream_sid
    assert first.snapshot.call_sid != second.snapshot.call_sid

    await first.close()
    await second.close()


async def test_call_session_resamples_and_writes_caller_audio_to_nova() -> None:
    nova = FakeNovaTransport()
    session = _session(nova=nova, queue_max_frames=4)
    session.handle_event(_start_event())

    session.handle_event(_media_event(sequence_number=2, chunk=1))
    session.handle_event(_media_event(sequence_number=3, chunk=2))
    await asyncio.wait_for(nova.audio_received.wait(), timeout=1)

    assert nova.sent_audio
    assert all(len(chunk) % 2 == 0 for chunk in nova.sent_audio)
    assert len(nova.sent_audio[0]) > 320

    await session.close()


async def test_call_session_converts_nova_audio_to_raw_twilio_mulaw() -> None:
    nova = FakeNovaTransport()
    session = _session(nova=nova, queue_max_frames=4)
    session.handle_event(_start_event())

    await nova.output_events.put(OutputAudio(pcm16le_24khz=b"\x01\x00" * 480))
    outbound = await asyncio.wait_for(session.next_outbound_message(), timeout=1)

    assert outbound is not None
    raw_mulaw = base64.b64decode(outbound.payload, validate=True)
    assert outbound.stream_sid == "MZ00000000000000000000000000000000"
    assert outbound.to_json() == {
        "event": "media",
        "streamSid": "MZ00000000000000000000000000000000",
        "media": {"payload": outbound.payload},
    }
    assert len(raw_mulaw) == 160
    assert raw_mulaw[:4] != b"RIFF"

    session.record_outbound_sent(outbound)
    await session.close()


async def test_call_session_rejects_odd_length_nova_pcm() -> None:
    nova = FakeNovaTransport()
    session = _session(nova=nova, queue_max_frames=4)
    session.handle_event(_start_event())

    await nova.output_events.put(OutputAudio(pcm16le_24khz=b"\x00"))
    await _wait_for_failure(session)

    assert session.failure_code == "NOVA_OUTPUT_AUDIO_INVALID"
    assert session.state is CallSessionState.FAILED
    assert session.snapshot.termination_reason is CallTerminationReason.NOVA_ERROR
    await session.close()


async def test_call_session_outbound_queue_overflow_fails_call() -> None:
    nova = FakeNovaTransport()
    session = _session(nova=nova, queue_max_frames=4, outbound_queue_max_frames=1)
    session.handle_event(_start_event())

    await nova.output_events.put(OutputAudio(pcm16le_24khz=b"\x01\x00" * 480))
    await nova.output_events.put(OutputAudio(pcm16le_24khz=b"\x02\x00" * 480))
    await _wait_for_failure(session)

    assert session.failure_code == "TWILIO_OUTBOUND_QUEUE_OVERFLOW"
    assert session.state is CallSessionState.FAILED
    assert session.snapshot.termination_reason is CallTerminationReason.QUEUE_OVERFLOW
    await session.close()


async def test_interruption_discards_stale_audio_before_new_generation() -> None:
    nova = FakeNovaTransport()
    telemetry = FakeTelemetry()
    session = _session(
        nova=nova,
        queue_max_frames=4,
        outbound_queue_max_frames=4,
        telemetry=telemetry,
    )
    session.handle_event(_start_event())
    await asyncio.wait_for(nova.started.wait(), timeout=1)

    await nova.output_events.put(OutputAudio(pcm16le_24khz=b"\x01\x00" * 480))
    await nova.output_events.put(OutputAudio(pcm16le_24khz=b"\x02\x00" * 480))
    await nova.output_events.put(InterruptionStarted(source_event_id="interruption-1"))
    await nova.output_events.put(OutputAudio(pcm16le_24khz=b"\x03\x00" * 480))
    await asyncio.sleep(0)

    clear = await asyncio.wait_for(session.next_outbound_message(), timeout=1)
    new_media = await asyncio.wait_for(session.next_outbound_message(), timeout=1)

    assert isinstance(clear, TwilioClearCommand)
    assert clear.generation == 1
    assert isinstance(new_media, TwilioMediaCommand)
    assert new_media.generation == 1
    assert len(base64.b64decode(new_media.payload, validate=True)) == 160
    assert [metric.name for metric in telemetry.metrics].count(MetricName.BARGE_INS) == 1
    await session.close()


async def test_first_completion_end_keeps_multi_turn_session_active() -> None:
    nova = FakeNovaTransport()
    session = _session(nova=nova, queue_max_frames=4)
    session.handle_event(_start_event())
    await asyncio.wait_for(nova.started.wait(), timeout=1)

    await nova.output_events.put(OutputAudio(pcm16le_24khz=b"\x01\x00" * 480))
    await nova.output_events.put(CompletionEnded())
    first_media = await asyncio.wait_for(session.next_outbound_message(), timeout=1)
    first_mark = await asyncio.wait_for(session.next_outbound_message(), timeout=1)

    assert isinstance(first_media, TwilioMediaCommand)
    assert isinstance(first_mark, TwilioMarkCommand)
    assert first_mark.name == "response-1"
    assert session.state is CallSessionState.ACTIVE
    assert session.failure_code is None
    assert not nova.events_cancelled.is_set()

    await nova.output_events.put(OutputAudio(pcm16le_24khz=b"\x02\x00" * 480))
    await nova.output_events.put(CompletionEnded())
    second_media = await asyncio.wait_for(session.next_outbound_message(), timeout=1)
    second_mark = await asyncio.wait_for(session.next_outbound_message(), timeout=1)

    assert isinstance(second_media, TwilioMediaCommand)
    assert isinstance(second_mark, TwilioMarkCommand)
    assert second_mark.name == "response-2"
    assert session.state is CallSessionState.ACTIVE
    assert session.failure_code is None
    await session.close()


async def test_completion_mark_is_bounded_and_acknowledged_after_clear() -> None:
    nova = FakeNovaTransport()
    session = _session(nova=nova, queue_max_frames=4)
    session.handle_event(_start_event())
    await asyncio.wait_for(nova.started.wait(), timeout=1)

    await nova.output_events.put(OutputAudio(pcm16le_24khz=b"\x01\x00" * 480))
    await nova.output_events.put(CompletionEnded())
    media = await asyncio.wait_for(session.next_outbound_message(), timeout=1)
    mark = await asyncio.wait_for(session.next_outbound_message(), timeout=1)

    assert isinstance(media, TwilioMediaCommand)
    assert isinstance(mark, TwilioMarkCommand)
    assert mark.name == "response-1"
    session.record_outbound_sent(media)
    session.record_outbound_sent(mark)

    await nova.output_events.put(InterruptionStarted(source_event_id="interruption-2"))
    clear = await asyncio.wait_for(session.next_outbound_message(), timeout=1)
    assert isinstance(clear, TwilioClearCommand)
    session.record_outbound_sent(clear)
    session.handle_event(MarkEvent(sequence_number=3, name=mark.name))

    assert session.failure_code is None
    await session.close()


async def test_continuation_events_emit_bounded_metrics_and_controlled_failure() -> None:
    nova = FakeNovaTransport()
    telemetry = FakeTelemetry()
    session = _session(nova=nova, queue_max_frames=4, telemetry=telemetry)
    session.handle_event(_start_event())
    await asyncio.wait_for(nova.started.wait(), timeout=1)

    await nova.output_events.put(ContinuationStarted(generation=1, history_turns=4))
    await nova.output_events.put(
        ContinuationSucceeded(
            generation=1,
            history_turns=4,
            buffered_pcm16_bytes=32_000,
            retired_cleanup_failed=False,
        )
    )
    await asyncio.sleep(0)

    names = [metric.name for metric in telemetry.metrics]
    assert names.count(MetricName.CONTINUATION_ATTEMPTS) == 1
    assert names.count(MetricName.CONTINUATION_SUCCESSES) == 1
    assert names.count(MetricName.CONTINUATION_FAILURES) == 0

    await nova.output_events.put(ContinuationFailed(generation=2, phase="STARTUP"))
    await _wait_for_failure(session)

    assert session.failure_code == "NOVA_CONTINUATION_FAILED"
    names = [metric.name for metric in telemetry.metrics]
    assert names.count(MetricName.CONTINUATION_FAILURES) == 1
    await session.close()


async def test_interruption_replaces_a_full_outbound_buffer_with_clear() -> None:
    nova = FakeNovaTransport()
    session = _session(
        nova=nova,
        queue_max_frames=4,
        outbound_queue_max_frames=1,
    )
    session.handle_event(_start_event())
    await asyncio.wait_for(nova.started.wait(), timeout=1)
    await nova.output_events.put(OutputAudio(pcm16le_24khz=b"\x01\x00" * 480))
    await nova.output_events.put(InterruptionStarted(source_event_id="interruption-3"))
    await asyncio.sleep(0)

    clear = await asyncio.wait_for(session.next_outbound_message(), timeout=1)

    assert isinstance(clear, TwilioClearCommand)
    assert session.failure_code is None
    await session.close()


async def test_call_session_queue_overflow_fails_instead_of_growing_latency() -> None:
    nova = FakeNovaTransport(block_start=True)
    session = _session(nova=nova, queue_max_frames=1)
    session.handle_event(_start_event())
    session.handle_event(_media_event(sequence_number=2, chunk=1))

    with pytest.raises(CallSessionError) as captured:
        session.handle_event(_media_event(sequence_number=3, chunk=2))

    assert captured.value.code == "TWILIO_AUDIO_QUEUE_OVERFLOW"
    assert session.state is CallSessionState.FAILED
    assert session.snapshot.termination_reason is CallTerminationReason.QUEUE_OVERFLOW
    await session.close()


async def test_call_session_cleanup_is_idempotent_under_concurrent_calls() -> None:
    nova = FakeNovaTransport(block_close=True)
    session = _session(nova=nova, queue_max_frames=4)
    session.handle_event(_start_event())

    close_tasks = [asyncio.create_task(session.close()) for _ in range(3)]
    await asyncio.wait_for(nova.close_started.wait(), timeout=1)
    nova.allow_close()
    await asyncio.gather(*close_tasks)

    assert nova.close_calls == 1
    assert session.closed is True
    assert session.state is CallSessionState.DISCONNECTED
    assert session.snapshot.termination_reason is CallTerminationReason.APPLICATION_SHUTDOWN
    assert session.snapshot.cleanup_error_code is None


async def test_call_session_cleanup_timeout_is_bounded_and_recorded() -> None:
    nova = FakeNovaTransport(block_close=True)
    session = _session(
        nova=nova,
        queue_max_frames=4,
        cleanup_timeout_seconds=0.01,
    )
    session.handle_event(_start_event())

    await asyncio.wait_for(session.close(), timeout=0.5)

    assert nova.close_calls == 1
    assert session.closed is True
    assert session.snapshot.cleanup_error_code == "NOVA_CLOSE_TIMEOUT"


@pytest.mark.asyncio
async def test_call_session_does_not_block_on_uncooperative_nova_close() -> None:
    nova = FakeNovaTransport(
        block_close=True,
        close_cancel_delay_seconds=0.1,
    )
    repository = FakeSessionRepository()
    session = _session(
        nova=nova,
        queue_max_frames=4,
        cleanup_timeout_seconds=0.01,
        repository=repository,
    )
    session.handle_event(_start_event())
    await asyncio.wait_for(nova.started.wait(), timeout=1)
    session.handle_event(StopEvent(sequence_number=2))

    await asyncio.wait_for(session.close(), timeout=0.1)

    assert session.closed is True
    assert session.snapshot.cleanup_error_code == "NOVA_CLOSE_TIMEOUT"
    assert repository.terminals[0].status is PersistedSessionStatus.COMPLETED
    await asyncio.wait_for(nova.close_finished.wait(), timeout=1)


@pytest.mark.asyncio
async def test_call_session_does_not_block_on_uncooperative_nova_worker() -> None:
    nova = FakeNovaTransport(
        end_events_on_close=False,
        event_cancel_delay_seconds=0.1,
    )
    repository = FakeSessionRepository()
    session = _session(
        nova=nova,
        queue_max_frames=4,
        cleanup_timeout_seconds=0.01,
        repository=repository,
    )
    session.handle_event(_start_event())
    await asyncio.wait_for(nova.started.wait(), timeout=1)
    session.handle_event(StopEvent(sequence_number=2))

    await asyncio.wait_for(session.close(), timeout=0.1)

    assert session.closed is True
    assert session.snapshot.cleanup_error_code == "NOVA_WORKER_CLOSE_TIMEOUT"
    assert repository.terminals[0].status is PersistedSessionStatus.COMPLETED
    await asyncio.wait_for(nova.events_cancelled.wait(), timeout=1)


async def test_call_session_persists_persona_lifecycle_and_final_turn_order() -> None:
    nova = FakeNovaTransport()
    repository = FakeSessionRepository()
    session = _session(
        nova=nova,
        queue_max_frames=4,
        repository=repository,
    )

    session.handle_event(_start_event())
    await nova.output_events.put(
        FinalTranscript(
            speaker=TranscriptSpeaker.CALLER,
            text="What services do you provide?",
            source_event_id="caller-final-1",
        )
    )
    await nova.output_events.put(
        FinalTranscript(
            speaker=TranscriptSpeaker.AGENT,
            text="We provide primary and specialty care.",
            source_event_id="agent-final-1",
        )
    )
    await nova.output_events.put(
        FinalTranscript(
            speaker=TranscriptSpeaker.AGENT,
            text="duplicate must be ignored",
            source_event_id="agent-final-1",
        )
    )
    assert await asyncio.to_thread(repository.two_turns_written.wait, 1)
    session.handle_event(StopEvent(sequence_number=2))
    await session.close()

    assert nova.system_prompts == [_PERSONA.system_prompt]
    assert len(repository.sessions) == 1
    persisted = repository.sessions[0]
    assert persisted.persona == _PERSONA
    assert persisted.call_sid == "CA" + ("0" * 32)
    assert len(repository.activations) == 1
    assert [turn.turn_number for turn in repository.turns] == [1, 2]
    assert [turn.speaker for turn in repository.turns] == [
        TranscriptSpeaker.CALLER,
        TranscriptSpeaker.AGENT,
    ]
    assert [turn.text for turn in repository.turns] == [
        "What services do you provide?",
        "We provide primary and specialty care.",
    ]
    assert len(repository.terminals) == 1
    terminal = repository.terminals[0]
    assert terminal.status.value == "COMPLETED"
    assert terminal.caller_turns == 1
    assert terminal.agent_turns == 1


async def test_successful_call_emits_expected_metrics_once() -> None:
    nova = FakeNovaTransport()
    repository = FakeSessionRepository()
    telemetry = FakeTelemetry()
    session = _session(
        nova=nova,
        queue_max_frames=4,
        repository=repository,
        telemetry=telemetry,
    )
    session.handle_event(_start_event())
    await nova.output_events.put(OutputAudio(pcm16le_24khz=b"\x01\x00" * 480))
    await nova.output_events.put(
        FinalTranscript(
            speaker=TranscriptSpeaker.CALLER,
            text="First final turn",
            source_event_id="caller-final-metric",
        )
    )
    await nova.output_events.put(
        FinalTranscript(
            speaker=TranscriptSpeaker.AGENT,
            text="Second final turn",
            source_event_id="agent-final-metric",
        )
    )
    assert await asyncio.to_thread(repository.two_turns_written.wait, 1)
    session.handle_event(StopEvent(sequence_number=2))

    await session.close()

    names = [metric.name for metric in telemetry.metrics]
    assert names.count(MetricName.CALLS_STARTED) == 1
    assert names.count(MetricName.CALL_START_TO_FIRST_AUDIO_MS) == 1
    assert names.count(MetricName.CALLS_COMPLETED) == 1
    assert names.count(MetricName.CALL_DURATION_MS) == 1
    assert names.count(MetricName.TRANSCRIPT_TURNS_PERSISTED) == 2
    assert MetricName.CALLS_FAILED not in names


async def test_cleanup_drains_final_transcript_emitted_while_nova_closes() -> None:
    final = FinalTranscript(
        speaker=TranscriptSpeaker.CALLER,
        text="A final phrase before hangup.",
        source_event_id="caller-close-final",
    )
    nova = FakeNovaTransport(events_on_close=[final])
    repository = FakeSessionRepository()
    session = _session(
        nova=nova,
        queue_max_frames=4,
        repository=repository,
    )
    session.handle_event(_start_event())
    await asyncio.sleep(0)
    session.handle_event(StopEvent(sequence_number=2))

    await session.close()

    assert [turn.text for turn in repository.turns] == ["A final phrase before hangup."]
    assert repository.terminals[0].caller_turns == 1


async def test_persistence_write_failure_terminates_only_that_call() -> None:
    repository = FakeSessionRepository(create_error=RuntimeError("synthetic persistence failure"))
    session = _session(
        nova=FakeNovaTransport(),
        queue_max_frames=4,
        repository=repository,
    )
    session.handle_event(_start_event())

    await _wait_for_failure(session)

    assert session.snapshot.state is CallSessionState.FAILED
    assert session.snapshot.failure_code == "PERSISTENCE_WRITE_FAILED"
    assert session.snapshot.termination_reason is CallTerminationReason.PERSISTENCE_ERROR
    await session.close()


async def test_persistence_queue_overflow_fails_instead_of_dropping_turns() -> None:
    session = _session(
        nova=FakeNovaTransport(),
        queue_max_frames=4,
        persistence_queue_max_events=1,
    )

    with pytest.raises(CallSessionError) as captured:
        session.handle_event(_start_event())

    assert captured.value.code == "PERSISTENCE_QUEUE_OVERFLOW"
    assert session.snapshot.state is CallSessionState.FAILED
    assert session.snapshot.termination_reason is CallTerminationReason.PERSISTENCE_ERROR
    await session.close()


@pytest.mark.asyncio
async def test_retryable_persistence_rejections_recover_in_fifo_order() -> None:
    def retryable() -> PersistenceError:
        return PersistenceError(
            "SESSION_CREATE_FAILED:ThrottlingException",
            code="ThrottlingException",
            retryable=True,
        )

    repository = FakeSessionRepository(create_errors=[retryable(), retryable()])
    telemetry = FakeTelemetry()
    session = _session(
        nova=FakeNovaTransport(),
        queue_max_frames=4,
        repository=repository,
        telemetry=telemetry,
        persistence_max_attempts=3,
        persistence_retry_base_delay_seconds=0.001,
    )

    session.handle_event(_start_event())
    await asyncio.wait_for(asyncio.to_thread(repository.session_created.wait), timeout=1)
    session.handle_event(StopEvent(sequence_number=2))
    await session.close()

    assert repository.create_calls == 3
    assert len(repository.sessions) == 1
    assert len(repository.activations) == 1
    assert session.snapshot.state is CallSessionState.COMPLETED
    retries = [
        metric for metric in telemetry.metrics if metric.name is MetricName.PERSISTENCE_RETRIES
    ]
    assert len(retries) == 2
    assert all(
        {(dimension.name, dimension.value) for dimension in metric.dimensions}
        == {("Environment", "test"), ("Component", "Persistence")}
        for metric in retries
    )


@pytest.mark.asyncio
async def test_retryable_persistence_rejections_fail_after_bounded_attempts() -> None:
    errors = [
        PersistenceError(
            "SESSION_CREATE_FAILED:ProvisionedThroughputExceededException",
            code="ProvisionedThroughputExceededException",
            retryable=True,
        )
        for _ in range(3)
    ]
    repository = FakeSessionRepository(create_errors=errors)
    telemetry = FakeTelemetry()
    session = _session(
        nova=FakeNovaTransport(),
        queue_max_frames=4,
        repository=repository,
        telemetry=telemetry,
        persistence_max_attempts=3,
        persistence_retry_base_delay_seconds=0.001,
    )

    session.handle_event(_start_event())
    await _wait_for_failure(session)
    await session.close()

    assert repository.create_calls == 3
    assert session.failure_code == "PERSISTENCE_WRITE_FAILED"
    assert sum(metric.name is MetricName.PERSISTENCE_RETRIES for metric in telemetry.metrics) == 2


@pytest.mark.asyncio
async def test_nonretryable_persistence_failure_is_not_retried() -> None:
    repository = FakeSessionRepository(
        create_error=PersistenceError(
            "SESSION_CREATE_FAILED:ValidationException",
            code="ValidationException",
        )
    )
    telemetry = FakeTelemetry()
    session = _session(
        nova=FakeNovaTransport(),
        queue_max_frames=4,
        repository=repository,
        telemetry=telemetry,
        persistence_max_attempts=3,
        persistence_retry_base_delay_seconds=0.001,
    )

    session.handle_event(_start_event())
    await _wait_for_failure(session)
    await session.close()

    assert repository.create_calls == 1
    assert session.failure_code == "PERSISTENCE_WRITE_FAILED"
    assert all(metric.name is not MetricName.PERSISTENCE_RETRIES for metric in telemetry.metrics)


def _session(
    *,
    nova: FakeNovaTransport,
    queue_max_frames: int,
    outbound_queue_max_frames: int = 4,
    persistence_queue_max_events: int = 100,
    repository: FakeSessionRepository | None = None,
    telemetry: TelemetryPublisher | None = None,
    clock: Callable[[], datetime] = _now,
    cleanup_timeout_seconds: float = 5.0,
    persistence_max_attempts: int = 3,
    persistence_retry_base_delay_seconds: float = 0.1,
    initial_text_prompt: str | None = None,
) -> CallSession:
    logger = cast(FilteringBoundLogger, structlog.get_logger())
    return CallSession(
        logger=logger,
        expected_twilio_account_sid="test-account-sid",
        malformed_frame_limit=1,
        audio_queue_max_frames=queue_max_frames,
        outbound_queue_max_frames=outbound_queue_max_frames,
        nova=nova,
        persona=_PERSONA,
        model_id="amazon.nova-2-sonic-v1:0",
        session_repository=repository or FakeSessionRepository(),
        persistence_queue_max_events=persistence_queue_max_events,
        transcript_retention_days=7,
        initial_text_prompt=initial_text_prompt,
        telemetry=telemetry,
        environment="test",
        clock=clock,
        cleanup_timeout_seconds=cleanup_timeout_seconds,
        persistence_max_attempts=persistence_max_attempts,
        persistence_retry_base_delay_seconds=persistence_retry_base_delay_seconds,
    )


async def _wait_for_failure(session: CallSession) -> None:
    with pytest.raises(CallSessionError):
        await asyncio.wait_for(session.wait_for_failure(), timeout=1)


def _start_event(*, stream_digit: str = "0") -> StartEvent:
    return StartEvent(
        sequence_number=1,
        stream_sid=f"MZ{stream_digit * 32}",
        call_sid=f"CA{stream_digit * 32}",
        account_sid="test-account-sid",
        media_format=MediaFormat(
            encoding="audio/x-mulaw",
            sample_rate=8_000,
            channels=1,
        ),
    )


def _media_event(*, sequence_number: int, chunk: int) -> MediaEvent:
    return MediaEvent(
        sequence_number=sequence_number,
        chunk=chunk,
        timestamp_ms=chunk * 20,
        payload=encode_twilio_mulaw_payload(b"\x00\x00" * 160),
        track="inbound",
    )
