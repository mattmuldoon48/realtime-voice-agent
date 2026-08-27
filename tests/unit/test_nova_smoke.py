"""Unit tests for the standalone smoke lifecycle without AWS or audio devices."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest

from realtime_voice_agent.config import NovaRuntimeConfig
from realtime_voice_agent.nova import smoke
from realtime_voice_agent.nova.events import (
    CompletionEnded,
    NovaServerEvent,
    NovaSessionState,
    OutputAudio,
)
from realtime_voice_agent.nova.smoke import (
    AwsIdentityError,
    AwsIdentitySummary,
    SmokeTestError,
    classify_aws_identity,
    run_smoke_test,
    safe_error_details,
)


class FakeLogger:
    def __init__(self) -> None:
        self.records: list[tuple[str, dict[str, object]]] = []

    def bind(self, **values: object) -> FakeLogger:
        self.records.append(("bound", values))
        return self

    def info(self, event: str, **values: object) -> None:
        self.records.append((event, values))

    def error(self, event: str, **values: object) -> None:
        self.records.append((event, values))


class FakeTransport:
    def __init__(self) -> None:
        self._state = NovaSessionState.NEW
        self.audio_sent = asyncio.Event()
        self.closed = False

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

    async def start_audio_input(self) -> None:
        assert self._state is NovaSessionState.ACTIVE

    async def send_audio(self, pcm16le_16khz: bytes) -> None:
        assert pcm16le_16khz == b"\x00\x00"
        self.audio_sent.set()

    async def events(self) -> AsyncIterator[NovaServerEvent]:
        await self.audio_sent.wait()
        yield OutputAudio(pcm16le_24khz=b"\x01\x00")
        yield CompletionEnded()

    async def finish_input(self) -> None:
        return None

    async def close(self) -> None:
        self.closed = True
        self._state = NovaSessionState.CLOSED


class FakeAudio:
    def __init__(self) -> None:
        self.opened = False
        self.played: list[bytes] = []
        self.closed = False

    async def open(self) -> None:
        self.opened = True

    async def read_input(self) -> bytes:
        await asyncio.sleep(0)
        return b"\x00\x00"

    async def play_output(self, pcm16le_24khz: bytes) -> None:
        self.played.append(pcm16le_24khz)

    async def close(self) -> None:
        self.closed = True


@pytest.fixture
def config() -> NovaRuntimeConfig:
    return NovaRuntimeConfig(
        region="us-west-2",
        profile="nova-demo",
        model_id="amazon.nova-2-sonic-v1:0",
        voice_id="matthew",
        input_sample_rate=16_000,
        output_sample_rate=24_000,
        channels=1,
        sample_width_bytes=2,
        chunk_frames=512,
    )


@pytest.mark.asyncio
async def test_smoke_lifecycle_emits_required_logs_and_cleans_up(
    monkeypatch: pytest.MonkeyPatch,
    config: NovaRuntimeConfig,
) -> None:
    async def fake_identity(_config: NovaRuntimeConfig) -> AwsIdentitySummary:
        return AwsIdentitySummary(identity_type="assumed-role")

    monkeypatch.setattr(smoke, "validate_aws_identity", fake_identity)
    transport = FakeTransport()
    audio = FakeAudio()
    logger = FakeLogger()

    await run_smoke_test(
        config=config,
        environment="test",
        timeout_seconds=1,
        logger=logger,  # type: ignore[arg-type]
        transport=transport,
        audio=audio,  # type: ignore[arg-type]
    )

    event_names = [event for event, _values in logger.records]
    assert "session_started" in event_names
    assert "connection_established" in event_names
    assert "first_audio_sent" in event_names
    assert "first_response_audio_received" in event_names
    assert "first_response_latency" in event_names
    assert "session_completed" in event_names
    assert audio.played == [b"\x01\x00"]
    assert audio.closed
    assert transport.closed


class NoResponseTransport(FakeTransport):
    async def events(self) -> AsyncIterator[NovaServerEvent]:
        await self.audio_sent.wait()
        yield CompletionEnded()


@pytest.mark.asyncio
async def test_completion_without_response_audio_fails_the_smoke_contract(
    monkeypatch: pytest.MonkeyPatch,
    config: NovaRuntimeConfig,
) -> None:
    async def fake_identity(_config: NovaRuntimeConfig) -> AwsIdentitySummary:
        return AwsIdentitySummary(identity_type="user")

    monkeypatch.setattr(smoke, "validate_aws_identity", fake_identity)
    transport = NoResponseTransport()
    audio = FakeAudio()

    with pytest.raises(SmokeTestError, match="before Nova returned response audio"):
        await run_smoke_test(
            config=config,
            environment="test",
            timeout_seconds=1,
            logger=FakeLogger(),  # type: ignore[arg-type]
            transport=transport,
            audio=audio,  # type: ignore[arg-type]
        )

    assert transport.closed
    assert audio.closed


def test_identity_classification_does_not_require_retaining_full_arn() -> None:
    assert classify_aws_identity("arn:aws:iam::123456789012:root") == "root"
    assert (
        classify_aws_identity("arn:aws:sts::123456789012:assumed-role/nova-smoke/session")
        == "assumed-role"
    )
    assert classify_aws_identity("arn:aws:iam::123456789012:user/nova-smoke") == "user"


def test_controlled_errors_never_render_external_exception_text() -> None:
    error = AwsIdentityError("AWS_CREDENTIALS_ERROR", "safe action")

    assert safe_error_details(error) == ("AWS_CREDENTIALS_ERROR", "safe action")
    assert safe_error_details(RuntimeError("credential-shaped secret")) == (
        "UNEXPECTED_ERROR",
        "The standalone Nova smoke test failed unexpectedly",
    )


def test_log_records_do_not_contain_audio_or_conversation_content() -> None:
    logger = FakeLogger()
    logger.info("first_audio_sent", message="First PCM input audio sent to Nova")

    rendered = repr(logger.records)
    assert "pcm16le_16khz" not in rendered
    assert "conversation" not in rendered
    assert "credential-shaped" not in rendered
