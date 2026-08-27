"""Standalone microphone-to-Nova-to-speaker smoke test."""

from __future__ import annotations

import argparse
import asyncio
import signal
import sys
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, cast

import boto3  # type: ignore[import-untyped]
import pyaudio  # type: ignore[import-untyped]
from pydantic import ValidationError
from structlog.typing import FilteringBoundLogger

from realtime_voice_agent.config import AppSettings, ConfigurationError, NovaRuntimeConfig
from realtime_voice_agent.nova.aws_transport import AwsNovaSonicTransport
from realtime_voice_agent.nova.events import CompletionEnded, OutputAudio, validate_pcm16_chunk
from realtime_voice_agent.nova.transport import NovaSonicTransport
from realtime_voice_agent.observability.logging import configure_local_logging, get_logger

_SYSTEM_PROMPT: Final = (
    "You are a concise and friendly voice assistant. Answer the user's spoken question "
    "clearly in one or two sentences."
)


class SmokeTestError(RuntimeError):
    """Controlled smoke-test failure with a safe operator-facing message."""

    def __init__(self, code: str, safe_message: str) -> None:
        super().__init__(safe_message)
        self.code = code
        self.safe_message = safe_message


class AudioDeviceError(SmokeTestError):
    """Local microphone or speaker setup failed."""

    def __init__(self) -> None:
        super().__init__(
            "AUDIO_DEVICE_ERROR",
            "Unable to open or use the default 16 kHz input and 24 kHz output devices",
        )


class AwsIdentityError(SmokeTestError):
    """The selected standard-chain identity is absent, invalid, or unsafe."""


@dataclass(frozen=True, slots=True)
class AwsIdentitySummary:
    """Non-sensitive identity classification returned by the startup check."""

    identity_type: str


@dataclass(slots=True)
class FirstAudioTracker:
    """Emit first-audio lifecycle events once and compute first-response latency."""

    logger: FilteringBoundLogger
    first_audio_sent_at: float | None = None
    first_response_received: bool = False

    def audio_sent(self, *, now: float) -> None:
        """Record the first successful audio write to Nova."""
        if self.first_audio_sent_at is not None:
            return
        self.first_audio_sent_at = now
        self.logger.info("first_audio_sent", message="First PCM input audio sent to Nova")

    def response_received(self, *, now: float) -> None:
        """Record first output audio and latency without logging its contents."""
        if self.first_response_received:
            return
        self.first_response_received = True
        self.logger.info(
            "first_response_audio_received",
            message="First PCM response audio received from Nova",
        )
        if self.first_audio_sent_at is not None:
            self.logger.info(
                "first_response_latency",
                message="Measured first-response audio latency",
                elapsed_ms=round((now - self.first_audio_sent_at) * 1000),
            )


class PyAudioDuplex:
    """Minimal macOS microphone/speaker bridge using the official sample's PyAudio."""

    def __init__(self, config: NovaRuntimeConfig) -> None:
        self._config = config
        self._audio: Any | None = None
        self._input_stream: Any | None = None
        self._output_stream: Any | None = None
        self._close_lock = asyncio.Lock()

    async def open(self) -> None:
        """Open default input/output devices with Nova's exact PCM contracts."""
        try:
            self._audio = pyaudio.PyAudio()
            self._input_stream = await asyncio.to_thread(
                self._audio.open,
                format=pyaudio.paInt16,
                channels=self._config.channels,
                rate=self._config.input_sample_rate,
                input=True,
                frames_per_buffer=self._config.chunk_frames,
            )
            self._output_stream = await asyncio.to_thread(
                self._audio.open,
                format=pyaudio.paInt16,
                channels=self._config.channels,
                rate=self._config.output_sample_rate,
                output=True,
                frames_per_buffer=self._config.chunk_frames,
            )
        except Exception as error:
            await self.close()
            raise AudioDeviceError from error

    async def read_input(self) -> bytes:
        """Read one PCM16LE mono 16 kHz chunk from the default microphone."""
        if self._input_stream is None:
            raise AudioDeviceError
        try:
            chunk = await asyncio.to_thread(
                self._input_stream.read,
                self._config.chunk_frames,
                exception_on_overflow=False,
            )
        except Exception as error:
            raise AudioDeviceError from error
        if not isinstance(chunk, bytes):
            raise AudioDeviceError
        validate_pcm16_chunk(chunk)
        return chunk

    async def play_output(self, pcm16le_24khz: bytes) -> None:
        """Play one validated PCM16LE mono 24 kHz Nova response chunk."""
        if self._output_stream is None:
            raise AudioDeviceError
        validate_pcm16_chunk(pcm16le_24khz)
        try:
            await asyncio.to_thread(self._output_stream.write, pcm16le_24khz)
        except Exception as error:
            raise AudioDeviceError from error

    async def close(self) -> None:
        """Idempotently release every PortAudio resource, even after one close error."""
        async with self._close_lock:
            input_stream, self._input_stream = self._input_stream, None
            output_stream, self._output_stream = self._output_stream, None
            audio, self._audio = self._audio, None
            cleanup_failed = False

            for stream in (input_stream, output_stream):
                if stream is None:
                    continue
                try:
                    if stream.is_active():
                        await asyncio.to_thread(stream.stop_stream)
                    await asyncio.to_thread(stream.close)
                except Exception:
                    cleanup_failed = True
            if audio is not None:
                try:
                    await asyncio.to_thread(audio.terminate)
                except Exception:
                    cleanup_failed = True
            if cleanup_failed:
                raise AudioDeviceError


async def validate_aws_identity(config: NovaRuntimeConfig) -> AwsIdentitySummary:
    """Authenticate with STS through the selected standard AWS credential chain."""
    try:
        session = boto3.Session(profile_name=config.profile, region_name=config.region)
        sts = session.client("sts")
        raw_response = await asyncio.to_thread(sts.get_caller_identity)
    except Exception as error:
        raise AwsIdentityError(
            "AWS_CREDENTIALS_ERROR",
            "Unable to authenticate with the selected AWS profile or credential chain",
        ) from error

    response = cast(Mapping[str, object], raw_response)
    arn = response.get("Arn")
    if not isinstance(arn, str):
        raise AwsIdentityError(
            "AWS_CREDENTIALS_ERROR",
            "AWS STS returned an identity without a valid ARN",
        )
    identity_type = classify_aws_identity(arn)
    if identity_type == "root":
        raise AwsIdentityError(
            "ROOT_AWS_IDENTITY",
            "Refusing to invoke Nova with the AWS root identity; configure a dedicated "
            "IAM identity",
        )
    return AwsIdentitySummary(identity_type=identity_type)


def classify_aws_identity(arn: str) -> str:
    """Classify an ARN without retaining or logging the complete identifier."""
    if arn.endswith(":root"):
        return "root"
    if ":assumed-role/" in arn:
        return "assumed-role"
    if ":role/" in arn:
        return "role"
    if ":user/" in arn:
        return "user"
    return "other"


async def run_smoke_test(
    *,
    config: NovaRuntimeConfig,
    environment: str,
    timeout_seconds: float,
    logger: FilteringBoundLogger,
    transport: NovaSonicTransport | None = None,
    audio: PyAudioDuplex | None = None,
) -> None:
    """Run one bounded, full-duplex spoken interaction."""
    if timeout_seconds <= 0:
        raise ConfigurationError("timeout must be greater than zero")

    session_id = str(uuid.uuid4())
    bound_logger = logger.bind(session_id=session_id, environment=environment)
    nova = transport or AwsNovaSonicTransport(config)
    duplex = audio or PyAudioDuplex(config)
    stop_event = asyncio.Event()
    first_audio = FirstAudioTracker(bound_logger)
    started_at = time.monotonic()
    tasks: list[asyncio.Task[None]] = []
    primary_error: BaseException | None = None
    remove_stop_handlers = _install_stop_handlers(stop_event)

    bound_logger.info(
        "session_started",
        message="Standalone Nova smoke session started",
        model_id=config.model_id,
        region=config.region,
        input_format="pcm16le_mono_16000hz",
        output_format="pcm16le_mono_24000hz",
    )

    try:
        async with asyncio.timeout(timeout_seconds):
            identity = await validate_aws_identity(config)
            await duplex.open()
            connection_started_at = time.monotonic()
            await nova.start(system_prompt=_SYSTEM_PROMPT)
            bound_logger.info(
                "connection_established",
                message="Nova bidirectional stream established",
                identity_type=identity.identity_type,
                elapsed_ms=round((time.monotonic() - connection_started_at) * 1000),
            )
            await nova.start_audio_input()

            tasks = [
                asyncio.create_task(
                    _capture_microphone(
                        transport=nova,
                        audio=duplex,
                        tracker=first_audio,
                        stop_event=stop_event,
                    ),
                    name="nova-smoke-capture",
                ),
                asyncio.create_task(
                    _play_nova_responses(
                        transport=nova,
                        audio=duplex,
                        tracker=first_audio,
                        stop_event=stop_event,
                    ),
                    name="nova-smoke-playback",
                ),
                asyncio.create_task(_wait_until_stopped(stop_event), name="nova-smoke-stop"),
            ]
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                task.result()
    except TimeoutError as error:
        primary_error = SmokeTestError(
            "SMOKE_TIMEOUT",
            "The standalone Nova smoke test exceeded its configured timeout",
        )
        raise primary_error from error
    except BaseException as error:
        primary_error = error
        raise
    finally:
        remove_stop_handlers()
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        cleanup_error: Exception | None = None
        try:
            await nova.close()
        except Exception as error:
            cleanup_error = error
        try:
            await duplex.close()
        except Exception as error:
            cleanup_error = cleanup_error or error

        if cleanup_error is not None:
            code, message = safe_error_details(cleanup_error)
            bound_logger.error(
                "error",
                message=message,
                error_code=code,
                error_type=type(cleanup_error).__name__,
                cleanup=True,
            )
            if primary_error is None:
                raise cleanup_error

    if first_audio.first_audio_sent_at is None:
        raise SmokeTestError(
            "NO_INPUT_AUDIO",
            "The smoke session ended before any microphone audio reached Nova",
        )
    if not first_audio.first_response_received:
        raise SmokeTestError(
            "NO_RESPONSE_AUDIO",
            "The smoke session ended before Nova returned response audio",
        )

    bound_logger.info(
        "session_completed",
        message="Standalone Nova smoke session completed",
        duration_ms=round((time.monotonic() - started_at) * 1000),
        response_audio_received=first_audio.first_response_received,
    )


async def _capture_microphone(
    *,
    transport: NovaSonicTransport,
    audio: PyAudioDuplex,
    tracker: FirstAudioTracker,
    stop_event: asyncio.Event,
) -> None:
    while not stop_event.is_set():
        chunk = await audio.read_input()
        await transport.send_audio(chunk)
        tracker.audio_sent(now=time.monotonic())


async def _play_nova_responses(
    *,
    transport: NovaSonicTransport,
    audio: PyAudioDuplex,
    tracker: FirstAudioTracker,
    stop_event: asyncio.Event,
) -> None:
    async for event in transport.events():
        if isinstance(event, OutputAudio):
            tracker.response_received(now=time.monotonic())
            await audio.play_output(event.pcm16le_24khz)
        elif isinstance(event, CompletionEnded):
            stop_event.set()
            return


async def _wait_until_stopped(stop_event: asyncio.Event) -> None:
    await stop_event.wait()


def _install_stop_handlers(stop_event: asyncio.Event) -> Callable[[], None]:
    """Install non-blocking Enter, SIGINT, and SIGTERM handling for macOS."""
    loop = asyncio.get_running_loop()
    installed_signals: list[signal.Signals] = []
    stdin_installed = False

    def request_stop() -> None:
        stop_event.set()

    for signal_number in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(signal_number, request_stop)
            installed_signals.append(signal_number)
        except (NotImplementedError, RuntimeError):
            continue

    if sys.stdin.isatty():
        try:
            loop.add_reader(sys.stdin.fileno(), request_stop)
            stdin_installed = True
        except (NotImplementedError, PermissionError):
            stdin_installed = False

    def remove() -> None:
        for signal_number in installed_signals:
            loop.remove_signal_handler(signal_number)
        if stdin_installed:
            loop.remove_reader(sys.stdin.fileno())

    return remove


def safe_error_details(error: BaseException) -> tuple[str, str]:
    """Map failures to controlled output without rendering external exception text."""
    if isinstance(error, SmokeTestError):
        return error.code, error.safe_message
    if isinstance(error, (ConfigurationError, ValidationError)):
        return "CONFIGURATION_ERROR", "Standalone Nova smoke configuration is invalid"

    error_type = type(error).__name__
    if error_type in {"NoCredentialsError", "ProfileNotFound", "IdentityChainError"}:
        return "AWS_CREDENTIALS_ERROR", "AWS credentials could not be resolved"
    if error_type in {"AccessDeniedException", "UnauthorizedException"}:
        return "AWS_ACCESS_DENIED", "The selected identity cannot invoke Nova 2 Sonic"
    if error_type == "ValidationException":
        return (
            "NOVA_VALIDATION_ERROR",
            "Nova rejected the model, Region, or event-stream configuration",
        )
    return "UNEXPECTED_ERROR", "The standalone Nova smoke test failed unexpectedly"


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the standalone smoke-test command-line interface."""
    parser = argparse.ArgumentParser(
        description="Run one local Amazon Nova 2 Sonic spoken smoke interaction"
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=120.0,
        help="Maximum total session duration (default: 120)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Load safe configuration, run the smoke test, and return a process code."""
    args = build_argument_parser().parse_args(argv)
    try:
        settings = AppSettings()
        config = settings.to_runtime_config()
    except (ConfigurationError, ValidationError) as error:
        configure_local_logging(level="INFO")
        logger = get_logger()
        code, message = safe_error_details(error)
        logger.error(
            "error",
            message=message,
            error_code=code,
            error_type=type(error).__name__,
        )
        return 2

    configure_local_logging(level=settings.log_level, service=settings.service_name)
    logger = get_logger(service=settings.service_name)
    try:
        asyncio.run(
            run_smoke_test(
                config=config,
                environment=settings.app_env,
                timeout_seconds=args.timeout_seconds,
                logger=logger,
            )
        )
    except BaseException as error:
        code, message = safe_error_details(error)
        logger.error(
            "error",
            message=message,
            error_code=code,
            error_type=type(error).__name__,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
