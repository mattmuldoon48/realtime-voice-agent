"""Per-call Twilio-to-Nova media session."""

from __future__ import annotations

import asyncio
import hashlib
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Final

from structlog.typing import FilteringBoundLogger

from realtime_voice_agent.audio.codecs import (
    TWILIO_SAMPLE_RATE,
    AudioCodecError,
    Pcm16MonoResampler,
    decode_twilio_mulaw_payload,
    encode_twilio_mulaw_payload,
)
from realtime_voice_agent.config import NOVA_INPUT_SAMPLE_RATE, NOVA_OUTPUT_SAMPLE_RATE
from realtime_voice_agent.nova.events import (
    CompletionEnded,
    ContinuationFailed,
    ContinuationStarted,
    ContinuationSucceeded,
    InterruptionStarted,
    NovaSessionState,
    OutputAudio,
)
from realtime_voice_agent.nova.transport import NovaSonicTransport
from realtime_voice_agent.observability.call_metrics import CallMetrics
from realtime_voice_agent.observability.models import (
    MetricComponent,
    MetricOutcome,
    NullTelemetryPublisher,
    TelemetryPublisher,
)
from realtime_voice_agent.persistence.errors import PersistenceError
from realtime_voice_agent.persistence.models import (
    PersistedSessionStatus,
    PersonaSnapshot,
    SessionStart,
    SessionTerminal,
    TranscriptTurn,
)
from realtime_voice_agent.persistence.ports import SessionRepository
from realtime_voice_agent.telephony.events import (
    ConnectedEvent,
    DtmfEvent,
    MarkEvent,
    MediaEvent,
    StartEvent,
    StopEvent,
    TwilioClearCommand,
    TwilioMarkCommand,
    TwilioMediaCommand,
    TwilioMediaStreamEvent,
    TwilioOutboundCommand,
    TwilioProtocolError,
)
from realtime_voice_agent.transcript import FinalTranscript, TranscriptSpeaker


class CallSessionError(RuntimeError):
    """Controlled terminal failure for one call."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class CallSessionState(StrEnum):
    """Validated lifecycle states for one telephone call."""

    STARTING = "STARTING"
    ACTIVE = "ACTIVE"
    COMPLETED = "COMPLETED"
    DISCONNECTED = "DISCONNECTED"
    FAILED = "FAILED"


class CallOutcome(StrEnum):
    """Bounded terminal outcome categories safe for persistence and metrics."""

    SUCCEEDED = "SUCCEEDED"
    DISCONNECTED = "DISCONNECTED"
    FAILED = "FAILED"


class CallTerminationReason(StrEnum):
    """Bounded reason categories for a call's first terminal transition."""

    TWILIO_STOP = "TWILIO_STOP"
    WEBSOCKET_DISCONNECT = "WEBSOCKET_DISCONNECT"
    PROTOCOL_ERROR = "PROTOCOL_ERROR"
    NOVA_ERROR = "NOVA_ERROR"
    QUEUE_OVERFLOW = "QUEUE_OVERFLOW"
    APPLICATION_SHUTDOWN = "APPLICATION_SHUTDOWN"
    PERSISTENCE_ERROR = "PERSISTENCE_ERROR"
    INTERNAL_ERROR = "INTERNAL_ERROR"
    DEMO_TIME_LIMIT = "DEMO_TIME_LIMIT"


@dataclass(frozen=True, slots=True)
class CallSessionSnapshot:
    """Immutable point-in-time lifecycle metadata for one call."""

    state: CallSessionState
    outcome: CallOutcome | None
    termination_reason: CallTerminationReason | None
    started_at: datetime
    activated_at: datetime | None
    ended_at: datetime | None
    stream_sid: str | None
    call_sid: str | None
    failure_code: str | None
    cleanup_error_code: str | None

    session_id: str
    persona: PersonaSnapshot


_TERMINAL_STATES: Final = frozenset(
    {
        CallSessionState.COMPLETED,
        CallSessionState.DISCONNECTED,
        CallSessionState.FAILED,
    }
)


def _utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True, slots=True)
class _MarkSessionActive:
    session_id: str
    activated_at: str


type _PersistenceCommand = (
    SessionStart | _MarkSessionActive | TranscriptTurn | SessionTerminal | None
)


def _persistence_operation(command: _PersistenceCommand) -> str:
    if isinstance(command, SessionStart):
        return "CREATE_SESSION"
    if isinstance(command, _MarkSessionActive):
        return "MARK_SESSION_ACTIVE"
    if isinstance(command, TranscriptTurn):
        return "APPEND_TRANSCRIPT_TURN"
    if isinstance(command, SessionTerminal):
        return "FINISH_SESSION"
    raise CallSessionError(
        "PERSISTENCE_COMMAND_INVALID",
        "Unsupported persistence command",
    )


class _OutboundCommandBuffer:
    """Bounded per-call Twilio commands with atomic playback interruption."""

    def __init__(self, max_commands: int) -> None:
        self._max_commands = max_commands
        self._generation = 0
        self._commands: deque[TwilioOutboundCommand] = deque()
        self._ready = asyncio.Event()
        self._closed = False

    @property
    def maxsize(self) -> int:
        return self._max_commands

    def put_nowait(self, command: TwilioOutboundCommand) -> bool:
        if self._closed or command.generation < self._generation:
            return False
        if command.generation > self._generation:
            raise ValueError("outbound command generation advanced without interruption")
        if len(self._commands) >= self._max_commands:
            raise asyncio.QueueFull
        self._commands.append(command)
        self._ready.set()
        return True

    def interrupt(self, command: TwilioClearCommand) -> None:
        if self._closed:
            return
        if command.generation <= self._generation:
            raise ValueError("playback interruption generation must advance")
        self._generation = command.generation
        self._commands.clear()
        self._commands.append(command)
        self._ready.set()

    async def get(self) -> TwilioOutboundCommand | None:
        while True:
            if self._commands:
                return self._commands.popleft()
            if self._closed:
                return None
            self._ready.clear()
            await self._ready.wait()

    def close(self) -> None:
        self._closed = True
        self._commands.clear()
        self._ready.set()


def _new_session_id() -> str:
    return str(uuid.uuid4())


class CallSession:
    """Own all mutable state for one Twilio-to-Nova call."""

    def __init__(
        self,
        *,
        logger: FilteringBoundLogger,
        expected_twilio_account_sid: str,
        malformed_frame_limit: int,
        audio_queue_max_frames: int,
        outbound_queue_max_frames: int,
        nova: NovaSonicTransport,
        persona: PersonaSnapshot,
        model_id: str,
        session_repository: SessionRepository,
        persistence_queue_max_events: int,
        transcript_retention_days: int,
        persist_transcripts: bool = True,
        telemetry: TelemetryPublisher | None = None,
        persistence_max_attempts: int = 3,
        persistence_retry_base_delay_seconds: float = 0.1,
        environment: str = "local",
        clock: Callable[[], datetime] = _utc_now,
        session_id_factory: Callable[[], str] = _new_session_id,
        cleanup_timeout_seconds: float = 5.0,
    ) -> None:
        if (
            audio_queue_max_frames <= 0
            or outbound_queue_max_frames <= 0
            or persistence_queue_max_events <= 0
        ):
            raise ValueError("queue bounds must be positive")
        if cleanup_timeout_seconds <= 0 or transcript_retention_days <= 0:
            raise ValueError("cleanup timeout and transcript retention must be positive")
        if persistence_max_attempts > 5:
            raise ValueError("persistence max attempts must be at most 5")
        if persistence_max_attempts <= 0 or not 0 < persistence_retry_base_delay_seconds <= 1:
            raise ValueError("persistence retry settings are outside their bounds")
        if not expected_twilio_account_sid:
            raise ValueError("expected Twilio account SID must not be blank")
        self._logger = logger
        self._expected_twilio_account_sid = expected_twilio_account_sid
        self._malformed_frame_limit = malformed_frame_limit
        self._nova = nova
        self._persona = persona
        self._model_id = model_id
        self._session_repository = session_repository
        self._clock = clock
        self._cleanup_timeout_seconds = cleanup_timeout_seconds
        self._persistence_max_attempts = persistence_max_attempts
        self._persistence_retry_base_delay_seconds = persistence_retry_base_delay_seconds
        self._persist_transcripts = persist_transcripts
        self._state = CallSessionState.STARTING
        self._outcome: CallOutcome | None = None
        self._termination_reason: CallTerminationReason | None = None
        self._started_at = self._timestamp()
        self._session_id = session_id_factory()
        if not self._session_id:
            raise ValueError("session ID must not be blank")
        self._logger = self._logger.bind(
            session_id=self._session_id,
            persona_id=self._persona.persona_id,
            persona_version=self._persona.version,
            nova_model_id=self._model_id,
        )
        self._call_metrics = CallMetrics(
            publisher=telemetry or NullTelemetryPublisher(),
            environment=environment,
        )
        self._expires_at = int(
            (self._started_at + timedelta(days=transcript_retention_days)).timestamp()
        )
        self._activated_at: datetime | None = None
        self._ended_at: datetime | None = None
        self._stream_sid: str | None = None
        self._call_sid: str | None = None
        self._inbound_media_frames = 0
        self._inbound_mulaw_bytes = 0
        self._inbound_pcm16_bytes = 0
        self._nova_input_pcm16_bytes = 0
        self._nova_input_frames = 0
        self._nova_response_audio_events = 0
        self._outbound_media_frames = 0
        self._outbound_mulaw_bytes = 0
        self._model_outbound_frames_sent = 0
        self._malformed_media_frames = 0
        self._caller_turns = 0
        self._agent_turns = 0
        self._next_turn_number = 1
        self._transcript_source_ids: set[str] = set()
        self._interruption_source_ids: set[str] = set()
        self._playback_generation = 0
        self._response_sequence = 0
        self._generation_has_audio = False
        self._pending_mark_names: set[str] = set()
        self._cleared_mark_names: set[str] = set()
        self._cleanup_complete = False
        self._failure_code: str | None = None
        self._persistence_error_code: str | None = None
        self._cleanup_error_code: str | None = None
        self._failure_event = asyncio.Event()
        self._nova_task: asyncio.Task[None] | None = None
        self._close_lock = asyncio.Lock()
        self._audio_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=audio_queue_max_frames)
        self._outbound_buffer = _OutboundCommandBuffer(outbound_queue_max_frames)
        self._persistence_task: asyncio.Task[None] | None = None
        self._persistence_queue: asyncio.Queue[_PersistenceCommand] = asyncio.Queue(
            maxsize=persistence_queue_max_events
        )
        self._inbound_resampler = Pcm16MonoResampler(
            source_rate_hz=TWILIO_SAMPLE_RATE,
            target_rate_hz=NOVA_INPUT_SAMPLE_RATE,
        )
        self._outbound_resampler = Pcm16MonoResampler(
            source_rate_hz=NOVA_OUTPUT_SAMPLE_RATE,
            target_rate_hz=TWILIO_SAMPLE_RATE,
        )

    @property
    def state(self) -> CallSessionState:
        """Return the current validated call lifecycle state."""
        return self._state

    @property
    def closed(self) -> bool:
        """Return whether this per-call session reached its first terminal state."""
        return self._state in _TERMINAL_STATES

    @property
    def failure_code(self) -> str | None:
        """Return the normalized terminal failure code, when present."""
        return self._failure_code

    @property
    def snapshot(self) -> CallSessionSnapshot:
        """Return immutable lifecycle metadata without mutable media state."""
        return CallSessionSnapshot(
            state=self._state,
            outcome=self._outcome,
            termination_reason=self._termination_reason,
            started_at=self._started_at,
            activated_at=self._activated_at,
            ended_at=self._ended_at,
            stream_sid=self._stream_sid,
            call_sid=self._call_sid,
            failure_code=self._failure_code,
            cleanup_error_code=self._cleanup_error_code,
            session_id=self._session_id,
            persona=self._persona,
        )

    async def wait_for_failure(self) -> None:
        """Raise when a background call task reaches a terminal failure."""
        await self._failure_event.wait()
        self._raise_if_failed()

    def disconnect(
        self,
        reason: CallTerminationReason = CallTerminationReason.WEBSOCKET_DISCONNECT,
    ) -> None:
        """Record the first non-failure disconnect outcome."""
        self._transition_terminal(
            state=CallSessionState.DISCONNECTED,
            outcome=CallOutcome.DISCONNECTED,
            reason=reason,
        )

    def fail(
        self,
        code: str,
        reason: CallTerminationReason | None = None,
    ) -> None:
        """Record the first normalized terminal failure."""
        self._set_failure(code, reason=reason)

    async def next_outbound_message(self) -> TwilioOutboundCommand | None:
        """Wait for the next encoded Twilio command or terminal sentinel."""
        message = await self._outbound_buffer.get()
        if message is None:
            self._raise_if_failed()
        return message

    def record_outbound_sent(self, message: TwilioOutboundCommand) -> None:
        """Record successful WebSocket delivery without retaining audio."""
        if message.stream_sid != self._stream_sid:
            raise CallSessionError(
                "TWILIO_OUTBOUND_STREAM_MISMATCH",
                "Outbound command does not match the active stream",
            )
        if isinstance(message, TwilioClearCommand):
            self._logger.info(
                "twilio_playback_clear_sent",
                playback_generation=message.generation,
            )
            return
        if isinstance(message, TwilioMarkCommand):
            if (
                message.name not in self._pending_mark_names
                and len(self._pending_mark_names) >= self._outbound_buffer.maxsize
            ):
                failure_code = "TWILIO_MARK_QUEUE_OVERFLOW"
                self._set_failure(failure_code)
                raise CallSessionError(failure_code, "Twilio mark state exceeded its call bound")
            self._pending_mark_names.add(message.name)
            self._logger.info(
                "twilio_playback_mark_sent",
                playback_generation=message.generation,
                pending_marks=len(self._pending_mark_names),
            )
            return
        self._outbound_media_frames += 1
        self._outbound_mulaw_bytes += message.mulaw_bytes
        self._model_outbound_frames_sent += 1
        if self._model_outbound_frames_sent == 1:
            self._logger.info(
                "twilio_first_model_audio_sent",
                outbound_payload_base64_bytes=len(message.payload),
                outbound_mulaw_bytes=message.mulaw_bytes,
            )

    def handle_event(self, event: TwilioMediaStreamEvent) -> None:
        """Apply one parsed Twilio event without blocking the media loop."""
        self._raise_if_failed()
        if self.closed:
            raise CallSessionError("CALL_SESSION_CLOSED", "Call session already closed")

        if isinstance(event, ConnectedEvent):
            self._logger.info(
                "twilio_connected",
                protocol=event.protocol,
                version=event.version,
                sequence_number=event.sequence_number,
            )
            return None
        if isinstance(event, StartEvent):
            self._handle_start(event)
            return
        if isinstance(event, MediaEvent):
            self._handle_media(event)
            return None
        if isinstance(event, DtmfEvent):
            self._logger.info(
                "twilio_dtmf_ignored",
                sequence_number=event.sequence_number,
                track=event.track,
            )
            return None
        if isinstance(event, MarkEvent):
            matched = event.name in self._pending_mark_names
            cleared = event.name in self._cleared_mark_names
            self._pending_mark_names.discard(event.name)
            self._cleared_mark_names.discard(event.name)
            self._logger.info(
                "twilio_playback_mark_received",
                sequence_number=event.sequence_number,
                matched=matched,
                playback_outcome="CLEARED" if cleared else "PLAYED" if matched else "UNKNOWN",
                pending_marks=len(self._pending_mark_names),
            )
            return None
        if isinstance(event, StopEvent):
            self._transition_terminal(
                state=CallSessionState.COMPLETED,
                outcome=CallOutcome.SUCCEEDED,
                reason=CallTerminationReason.TWILIO_STOP,
            )
            self._publish_outbound_end()
            self._logger.info(
                "twilio_stopped",
                sequence_number=event.sequence_number,
                inbound_media_frames=self._inbound_media_frames,
                inbound_mulaw_bytes=self._inbound_mulaw_bytes,
                inbound_pcm16_bytes=self._inbound_pcm16_bytes,
                nova_input_frames=self._nova_input_frames,
                nova_input_pcm16_bytes=self._nova_input_pcm16_bytes,
                nova_response_audio_events=self._nova_response_audio_events,
                outbound_media_frames=self._outbound_media_frames,
                outbound_mulaw_bytes=self._outbound_mulaw_bytes,
                malformed_media_frames=self._malformed_media_frames,
            )
            return None
        raise TwilioProtocolError("TWILIO_EVENT_UNHANDLED", "Unhandled Twilio event")

    async def close(self) -> None:
        """Idempotently terminalize the call and close Nova and persistence once."""
        async with self._close_lock:
            if self._cleanup_complete:
                return
            self.disconnect(CallTerminationReason.APPLICATION_SHUTDOWN)
            self._publish_outbound_end()
            nova_was_active = self._nova.state is NovaSessionState.ACTIVE
            nova_closed_cleanly = False
            nova_close_task = asyncio.create_task(
                self._nova.close(),
                name="nova-transport-close",
            )
            done, _ = await asyncio.wait(
                {nova_close_task},
                timeout=self._cleanup_timeout_seconds,
            )
            if not done:
                nova_close_task.cancel()
                self._cleanup_error_code = "NOVA_CLOSE_TIMEOUT"
                self._call_metrics.component_error(MetricComponent.NOVA)
                self._logger.error(
                    "nova_close_failed",
                    error_code=self._cleanup_error_code,
                )
            else:
                try:
                    nova_close_task.result()
                except Exception as error:
                    self._cleanup_error_code = "NOVA_CLOSE_FAILED"
                    self._call_metrics.component_error(MetricComponent.NOVA)
                    self._logger.error(
                        "nova_close_failed",
                        error_code=self._cleanup_error_code,
                        error_type=type(error).__name__,
                    )
                else:
                    nova_closed_cleanly = True
            if self._nova_task is not None and not self._nova_task.done():
                if nova_closed_cleanly and nova_was_active:
                    _, pending = await asyncio.wait(
                        {self._nova_task},
                        timeout=self._cleanup_timeout_seconds,
                    )
                    if pending and self._cleanup_error_code is None:
                        self._cleanup_error_code = "NOVA_WORKER_CLOSE_TIMEOUT"
                if not self._nova_task.done():
                    self._nova_task.cancel()
                    _, pending = await asyncio.wait(
                        {self._nova_task},
                        timeout=self._cleanup_timeout_seconds,
                    )
                    if pending:
                        self._logger.error(
                            "nova_worker_cancel_timed_out",
                            error_code="NOVA_WORKER_CANCEL_TIMEOUT",
                        )
            await self._close_persistence()
            self._cleanup_complete = True
            self._logger.info(
                "call_session_closed",
                lifecycle_state=self._state,
                outcome=self._outcome,
                termination_reason=self._termination_reason,
                started_at=self._started_at.isoformat(),
                activated_at=(
                    self._activated_at.isoformat() if self._activated_at is not None else None
                ),
                ended_at=self._ended_at.isoformat() if self._ended_at is not None else None,
                duration_ms=self._duration_ms(),
                inbound_media_frames=self._inbound_media_frames,
                nova_input_frames=self._nova_input_frames,
                nova_input_pcm16_bytes=self._nova_input_pcm16_bytes,
                nova_response_audio_events=self._nova_response_audio_events,
                caller_turns=self._caller_turns,
                agent_turns=self._agent_turns,
                failure_code=self._failure_code,
                cleanup_error_code=self._cleanup_error_code,
            )

    def _handle_start(self, event: StartEvent) -> None:
        if self._stream_sid is not None:
            raise TwilioProtocolError("TWILIO_DUPLICATE_START", "Duplicate start event")
        if event.account_sid != self._expected_twilio_account_sid:
            raise TwilioProtocolError(
                "TWILIO_ACCOUNT_SID_MISMATCH",
                "Twilio start account does not match configured account",
            )
        self._stream_sid = event.stream_sid
        self._call_sid = event.call_sid
        self._logger = self._logger.bind(
            stream_sid_hash=hash_identifier(event.stream_sid),
            call_sid_hash=hash_identifier(event.call_sid),
        )
        self._call_metrics.call_started()
        self._persistence_task = asyncio.create_task(
            self._run_persistence(),
            name=f"call-persistence-{hash_identifier(self._session_id)}",
        )
        self._enqueue_persistence(
            SessionStart(
                session_id=self._session_id,
                call_sid=event.call_sid,
                stream_sid=event.stream_sid,
                persona=self._persona,
                model_id=self._model_id,
                started_at=self._started_at,
                expires_at=self._expires_at,
            )
        )
        self._transition_active()
        if self._activated_at is None:
            raise CallSessionError(
                "CALL_ACTIVATION_TIMESTAMP_MISSING",
                "Active call is missing its activation timestamp",
            )
        self._enqueue_persistence(
            _MarkSessionActive(
                session_id=self._session_id,
                activated_at=self._activated_at.isoformat(),
            )
        )
        self._logger.info(
            "twilio_started",
            sequence_number=event.sequence_number,
            media_encoding=event.media_format.encoding,
            media_sample_rate=event.media_format.sample_rate,
            media_channels=event.media_format.channels,
        )
        self._nova_task = asyncio.create_task(
            self._run_nova(),
            name=f"nova-call-{hash_identifier(event.call_sid)}",
        )

    def _handle_media(self, event: MediaEvent) -> None:
        if self._stream_sid is None or self._call_sid is None:
            raise TwilioProtocolError("TWILIO_MEDIA_BEFORE_START", "Media arrived before start")
        try:
            pcm16_8khz = decode_twilio_mulaw_payload(event.payload)
            pcm16_16khz = self._inbound_resampler.convert(pcm16_8khz)
        except AudioCodecError as error:
            self._malformed_media_frames += 1
            self._logger.info(
                "twilio_malformed_media_frame",
                sequence_number=event.sequence_number,
                chunk=event.chunk,
                malformed_media_frames=self._malformed_media_frames,
            )
            if self._malformed_media_frames > self._malformed_frame_limit:
                raise TwilioProtocolError(
                    "TWILIO_MALFORMED_MEDIA_LIMIT", "Too many media errors"
                ) from error
            return

        try:
            self._audio_queue.put_nowait(pcm16_16khz)
        except asyncio.QueueFull as error:
            failure_code = "TWILIO_AUDIO_QUEUE_OVERFLOW"
            self._set_failure(failure_code)
            self._logger.error(
                "call_queue_overflow",
                error_code=failure_code,
                queue="twilio_audio_in",
                queue_max_frames=self._audio_queue.maxsize,
            )
            raise CallSessionError(
                failure_code,
                "Twilio audio queue exceeded its per-call bound",
            ) from error

        self._inbound_media_frames += 1
        self._inbound_mulaw_bytes += len(pcm16_8khz) // 2
        self._inbound_pcm16_bytes += len(pcm16_8khz)
        self._nova_input_pcm16_bytes += len(pcm16_16khz)
        self._logger.info(
            "twilio_inbound_media_queued",
            sequence_number=event.sequence_number,
            chunk=event.chunk,
            timestamp_ms=event.timestamp_ms,
            track=event.track,
            inbound_media_frames=self._inbound_media_frames,
            inbound_mulaw_bytes=self._inbound_mulaw_bytes,
            inbound_pcm16_bytes=self._inbound_pcm16_bytes,
            nova_input_pcm16_bytes=self._nova_input_pcm16_bytes,
            queue_depth_frames=self._audio_queue.qsize(),
        )

    async def _run_nova(self) -> None:
        try:
            await self._nova.start(system_prompt=self._persona.system_prompt)
            await self._nova.start_audio_input()
            self._logger.info("nova_input_bridge_started")
            await self._run_nova_workers()
        except asyncio.CancelledError:
            raise
        except CallSessionError as error:
            self._set_failure(error.code)
            self._publish_outbound_end()
        except Exception as error:
            failure_code = "NOVA_STREAM_FAILED"
            self._set_failure(failure_code)
            self._logger.error(
                "nova_stream_failed",
                error_code=failure_code,
                error_type=type(error).__name__,
            )
            self._publish_outbound_end()

    async def _run_nova_workers(self) -> None:
        tasks = (
            asyncio.create_task(self._write_nova_audio(), name="nova-input-writer"),
            asyncio.create_task(self._observe_nova_events(), name="nova-event-observer"),
        )
        try:
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                task.result()
            raise CallSessionError(
                "NOVA_STREAM_ENDED",
                "A Nova stream worker ended before call cleanup",
            )
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _write_nova_audio(self) -> None:
        while True:
            pcm16_16khz = await self._audio_queue.get()
            await self._nova.send_audio(pcm16_16khz)
            self._nova_input_frames += 1
            if self._nova_input_frames == 1:
                self._logger.info(
                    "nova_first_input_audio_sent",
                    pcm16_bytes=len(pcm16_16khz),
                )

    async def _observe_nova_events(self) -> None:
        async for event in self._nova.events():
            if isinstance(event, OutputAudio):
                try:
                    pcm16_8khz = self._outbound_resampler.convert(event.pcm16le_24khz)
                    payload = encode_twilio_mulaw_payload(pcm16_8khz)
                except AudioCodecError as error:
                    raise CallSessionError(
                        "NOVA_OUTPUT_AUDIO_INVALID",
                        "Nova returned invalid PCM16 response audio",
                    ) from error
                self._nova_response_audio_events += 1
                if self._nova_response_audio_events == 1:
                    self._logger.info(
                        "nova_first_response_audio_received",
                        pcm16_bytes=len(event.pcm16le_24khz),
                    )
                    self._call_metrics.call_start_to_first_audio(
                        elapsed_ms=int(
                            (self._timestamp() - self._started_at).total_seconds() * 1_000
                        )
                    )
                self._generation_has_audio = True
                self._enqueue_outbound(
                    TwilioMediaCommand(
                        stream_sid=self._require_stream_sid(),
                        payload=payload,
                        mulaw_bytes=len(pcm16_8khz) // 2,
                        generation=self._playback_generation,
                    )
                )
            elif isinstance(event, FinalTranscript):
                self._enqueue_final_transcript(event)
            elif isinstance(event, InterruptionStarted):
                self._handle_interruption(event)
            elif isinstance(event, CompletionEnded):
                self._enqueue_response_mark()
                self._logger.info("nova_completion_ended")
            elif isinstance(event, ContinuationStarted):
                self._call_metrics.continuation_attempted()
                self._logger.info(
                    "nova_continuation_started",
                    nova_generation=event.generation,
                    history_turns=event.history_turns,
                )
            elif isinstance(event, ContinuationSucceeded):
                self._call_metrics.continuation_succeeded()
                if event.retired_cleanup_failed:
                    self._call_metrics.continuation_failed()
                self._logger.info(
                    "nova_continuation_succeeded",
                    nova_generation=event.generation,
                    history_turns=event.history_turns,
                    buffered_pcm16_bytes=event.buffered_pcm16_bytes,
                    retired_cleanup_failed=event.retired_cleanup_failed,
                )
            elif isinstance(event, ContinuationFailed):
                self._call_metrics.continuation_failed()
                self._logger.error(
                    "nova_continuation_failed",
                    error_code="NOVA_CONTINUATION_FAILED",
                    nova_generation=event.generation,
                    continuation_phase=event.phase,
                )
                raise CallSessionError(
                    "NOVA_CONTINUATION_FAILED",
                    "Nova replacement stream could not become active",
                )

    def _handle_interruption(self, event: InterruptionStarted) -> None:
        if event.source_event_id in self._interruption_source_ids:
            return
        self._interruption_source_ids.add(event.source_event_id)
        self._playback_generation += 1
        cleared_pending_marks = len(self._pending_mark_names)
        self._cleared_mark_names.update(self._pending_mark_names)
        self._generation_has_audio = False
        self._outbound_buffer.interrupt(
            TwilioClearCommand(
                stream_sid=self._require_stream_sid(),
                generation=self._playback_generation,
            )
        )
        self._call_metrics.barge_in()
        self._logger.info(
            "nova_playback_interrupted",
            playback_generation=self._playback_generation,
            cleared_pending_marks=cleared_pending_marks,
        )

    def _enqueue_response_mark(self) -> None:
        if not self._generation_has_audio:
            return
        self._generation_has_audio = False
        self._response_sequence += 1
        self._enqueue_outbound(
            TwilioMarkCommand(
                stream_sid=self._require_stream_sid(),
                name=f"response-{self._response_sequence}",
                generation=self._playback_generation,
            )
        )

    def _enqueue_final_transcript(self, event: FinalTranscript) -> None:
        if event.source_event_id in self._transcript_source_ids:
            return
        turn_number = self._next_turn_number
        if self._persist_transcripts:
            self._enqueue_persistence(
                TranscriptTurn(
                    session_id=self._session_id,
                    turn_number=turn_number,
                    speaker=event.speaker,
                    text=event.text,
                    source_event_id=event.source_event_id,
                    created_at=self._timestamp(),
                    expires_at=self._expires_at,
                )
            )
        self._transcript_source_ids.add(event.source_event_id)
        self._next_turn_number += 1
        if event.speaker is TranscriptSpeaker.CALLER:
            self._caller_turns += 1
        else:
            self._agent_turns += 1
        self._logger.info(
            "final_transcript_observed",
            turn_number=turn_number,
            speaker=event.speaker,
            persisted=self._persist_transcripts,
        )

    def _enqueue_persistence(self, command: _PersistenceCommand) -> None:
        try:
            self._persistence_queue.put_nowait(command)
        except asyncio.QueueFull as error:
            failure_code = "PERSISTENCE_QUEUE_OVERFLOW"
            self._set_failure(
                failure_code,
                reason=CallTerminationReason.PERSISTENCE_ERROR,
            )
            self._publish_outbound_end()
            self._logger.error(
                "call_queue_overflow",
                error_code=failure_code,
                queue="persistence_events",
                queue_max_events=self._persistence_queue.maxsize,
            )
            raise CallSessionError(
                failure_code,
                "Persistence queue exceeded its per-call bound",
            ) from error

    async def _run_persistence(self) -> None:
        try:
            while True:
                command = await self._persistence_queue.get()
                if command is None:
                    return
                await self._dispatch_persistence(command)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            self._persistence_error_code = "PERSISTENCE_WRITE_FAILED"
            self._set_failure(
                self._persistence_error_code,
                reason=CallTerminationReason.PERSISTENCE_ERROR,
            )
            self._publish_outbound_end()
            self._logger.error(
                "persistence_write_failed",
                error_code=self._persistence_error_code,
                error_type=type(error).__name__,
            )
            await self._attempt_terminal_persistence()

    async def _dispatch_persistence(self, command: _PersistenceCommand) -> None:
        operation = _persistence_operation(command)
        for attempt in range(1, self._persistence_max_attempts + 1):
            try:
                await self._dispatch_persistence_once(command)
            except PersistenceError as error:
                if not error.retryable or attempt >= self._persistence_max_attempts:
                    raise
                delay_seconds = min(
                    self._persistence_retry_base_delay_seconds * (2 ** (attempt - 1)),
                    1.0,
                )
                self._call_metrics.persistence_retry()
                self._logger.warning(
                    "persistence_retry_scheduled",
                    operation=operation,
                    error_code=error.code,
                    attempt=attempt,
                    next_attempt=attempt + 1,
                    max_attempts=self._persistence_max_attempts,
                    delay_ms=round(delay_seconds * 1_000),
                )
                await asyncio.sleep(delay_seconds)
            else:
                return

    async def _dispatch_persistence_once(self, command: _PersistenceCommand) -> None:
        if isinstance(command, SessionStart):
            await asyncio.to_thread(self._session_repository.create_session, command)
            return
        if isinstance(command, _MarkSessionActive):
            await asyncio.to_thread(
                self._session_repository.mark_session_active,
                command.session_id,
                command.activated_at,
            )
            return
        if isinstance(command, TranscriptTurn):
            await asyncio.to_thread(
                self._session_repository.append_transcript_turn,
                command,
            )
            self._call_metrics.transcript_turn_persisted()
            return
        if isinstance(command, SessionTerminal):
            await asyncio.to_thread(self._session_repository.finish_session, command)
            return
        raise CallSessionError(
            "PERSISTENCE_COMMAND_INVALID",
            "Unsupported persistence command",
        )

    async def _close_persistence(self) -> None:
        task = self._persistence_task
        if task is None:
            return
        if not task.done():
            try:
                async with asyncio.timeout(self._cleanup_timeout_seconds):
                    await self._persistence_queue.put(self._build_session_terminal())
                    await self._persistence_queue.put(None)
                    await asyncio.shield(task)
            except TimeoutError:
                self._cleanup_error_code = "PERSISTENCE_CLOSE_TIMEOUT"
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
                self._call_metrics.component_error(MetricComponent.PERSISTENCE)
                self._logger.error(
                    "persistence_close_failed",
                    error_code=self._cleanup_error_code,
                )
        else:
            await asyncio.gather(task, return_exceptions=True)
        if (
            self._persistence_error_code is not None
            and self._failure_code != self._persistence_error_code
            and self._cleanup_error_code is None
        ):
            self._cleanup_error_code = self._persistence_error_code

    async def _attempt_terminal_persistence(self) -> None:
        try:
            terminal = self._build_session_terminal()
            await asyncio.to_thread(self._session_repository.finish_session, terminal)
        except Exception:
            return

    def _build_session_terminal(self) -> SessionTerminal:
        if self._ended_at is None or self._outcome is None or self._termination_reason is None:
            raise CallSessionError(
                "CALL_TERMINAL_METADATA_MISSING",
                "Cannot persist a non-terminal call",
            )
        duration_ms = self._duration_ms()
        if duration_ms is None:
            raise CallSessionError(
                "CALL_TERMINAL_DURATION_MISSING",
                "Cannot persist a terminal call without a duration",
            )
        return SessionTerminal(
            session_id=self._session_id,
            status=PersistedSessionStatus(self._state.value),
            outcome=self._outcome.value,
            termination_reason=self._termination_reason.value,
            ended_at=self._ended_at,
            duration_ms=duration_ms,
            inbound_media_frames=self._inbound_media_frames,
            nova_input_frames=self._nova_input_frames,
            nova_input_pcm16_bytes=self._nova_input_pcm16_bytes,
            nova_response_audio_events=self._nova_response_audio_events,
            outbound_media_frames=self._outbound_media_frames,
            outbound_mulaw_bytes=self._outbound_mulaw_bytes,
            caller_turns=self._caller_turns,
            agent_turns=self._agent_turns,
            failure_code=self._failure_code,
            cleanup_error_code=self._cleanup_error_code,
        )

    def _enqueue_outbound(self, message: TwilioOutboundCommand) -> None:
        try:
            self._outbound_buffer.put_nowait(message)
        except asyncio.QueueFull as error:
            failure_code = "TWILIO_OUTBOUND_QUEUE_OVERFLOW"
            self._set_failure(failure_code)
            self._logger.error(
                "call_queue_overflow",
                error_code=failure_code,
                queue="twilio_audio_out",
                queue_max_frames=self._outbound_buffer.maxsize,
            )
            raise CallSessionError(
                failure_code,
                "Twilio outbound queue exceeded its per-call bound",
            ) from error

    def _require_stream_sid(self) -> str:
        if self._stream_sid is None:
            raise CallSessionError(
                "TWILIO_OUTPUT_BEFORE_START",
                "Nova audio arrived before the Twilio stream started",
            )
        return self._stream_sid

    def _publish_outbound_end(self) -> None:
        self._outbound_buffer.close()

    def _set_failure(
        self,
        code: str,
        *,
        reason: CallTerminationReason | None = None,
    ) -> None:
        transitioned = self._transition_terminal(
            state=CallSessionState.FAILED,
            outcome=CallOutcome.FAILED,
            reason=reason or _failure_reason(code),
            failure_code=code,
        )
        if transitioned:
            self._failure_event.set()

    def _transition_active(self) -> None:
        if self._state is not CallSessionState.STARTING:
            raise CallSessionError(
                "CALL_STATE_TRANSITION_INVALID",
                f"Call cannot become active from {self._state}",
            )
        self._state = CallSessionState.ACTIVE
        self._activated_at = self._timestamp()
        self._logger.info("call_session_active", lifecycle_state=self._state)

    def _transition_terminal(
        self,
        *,
        state: CallSessionState,
        outcome: CallOutcome,
        reason: CallTerminationReason,
        failure_code: str | None = None,
    ) -> bool:
        if state not in _TERMINAL_STATES:
            raise ValueError(f"{state} is not a terminal call state")
        if self.closed:
            return False
        if self._state not in {CallSessionState.STARTING, CallSessionState.ACTIVE}:
            raise CallSessionError(
                "CALL_STATE_TRANSITION_INVALID",
                f"Call cannot terminate from {self._state}",
            )
        self._state = state
        self._outcome = outcome
        self._termination_reason = reason
        self._failure_code = failure_code
        self._ended_at = self._timestamp()
        duration_ms = int((self._ended_at - self._started_at).total_seconds() * 1_000)
        self._call_metrics.call_terminal(
            outcome=MetricOutcome(outcome.value),
            duration_ms=duration_ms,
            failure_code=failure_code,
        )
        self._logger.info(
            "call_session_terminal",
            lifecycle_state=state,
            outcome=outcome,
            termination_reason=reason,
            failure_code=failure_code,
        )
        return True

    def _timestamp(self) -> datetime:
        timestamp = self._clock()
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError("CallSession clock must return a timezone-aware timestamp")
        return timestamp.astimezone(UTC)

    def _duration_ms(self) -> int | None:
        if self._ended_at is None:
            return None
        return int((self._ended_at - self._started_at).total_seconds() * 1_000)

    def _raise_if_failed(self) -> None:
        if self._failure_code is not None:
            raise CallSessionError(self._failure_code, "Call session failed")


def _failure_reason(code: str) -> CallTerminationReason:
    if "QUEUE_OVERFLOW" in code:
        return CallTerminationReason.QUEUE_OVERFLOW
    if code.startswith("PERSISTENCE_"):
        return CallTerminationReason.PERSISTENCE_ERROR
    if code.startswith("NOVA_"):
        return CallTerminationReason.NOVA_ERROR
    if code.startswith(("TWILIO_", "CALL_")):
        return CallTerminationReason.PROTOCOL_ERROR
    return CallTerminationReason.INTERNAL_ERROR


def hash_identifier(identifier: str) -> str:
    """Return a bounded one-way identifier fingerprint for safe logs."""
    return hashlib.sha256(identifier.encode("utf-8")).hexdigest()[:16]
