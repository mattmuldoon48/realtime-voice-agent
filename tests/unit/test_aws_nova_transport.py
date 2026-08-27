"""Unit tests for AWS adapter lifecycle paths that do not call AWS."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest
from smithy_aws_core.identity import ContainerCredentialsResolver

from realtime_voice_agent.config import NovaRuntimeConfig
from realtime_voice_agent.nova.aws_transport import (
    AwsNovaSonicTransport,
    _create_credentials_resolver,
)
from realtime_voice_agent.nova.events import (
    CompletionEnded,
    NovaProtocolError,
    NovaSessionState,
    OutputAudio,
)


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
async def test_ecs_task_role_credentials_do_not_depend_on_unregistered_chain_plugin(
    config: NovaRuntimeConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI", "/v2/credentials/task")

    resolver = await _create_credentials_resolver(config, cast(Any, object()))

    assert isinstance(resolver, ContainerCredentialsResolver)


@pytest.mark.asyncio
async def test_close_before_start_is_idempotent(config: NovaRuntimeConfig) -> None:
    transport = AwsNovaSonicTransport(config)

    await transport.close()
    await transport.close()

    assert transport.state is NovaSessionState.CLOSED


@pytest.mark.asyncio
async def test_interactive_text_must_precede_audio_input(
    config: NovaRuntimeConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = AwsNovaSonicTransport(config)
    transport._state = NovaSessionState.ACTIVE
    sent: list[bytes] = []

    async def capture(payload: bytes) -> None:
        sent.append(payload)

    monkeypatch.setattr(transport, "_send", capture)
    await transport.send_text("Hello")

    assert len(sent) == 3
    assert b'"interactive":true' in sent[0]
    assert b'"role":"USER"' in sent[0]
    assert b'"content":"Hello"' in sent[1]

    transport._audio_started = True
    with pytest.raises(NovaProtocolError, match="must precede audio input"):
        await transport.send_text("Hello again")


@pytest.mark.asyncio
async def test_events_continue_across_multiple_response_cycles(
    config: NovaRuntimeConfig,
) -> None:
    class FakeStream:
        def __init__(self) -> None:
            self.await_output_calls = 0
            self.payloads = iter(
                (
                    b'{"event":{"audioOutput":{"content":"AAA="}}}',
                    b'{"event":{"completionEnd":{}}}',
                    b'{"event":{"audioOutput":{"content":"AAA="}}}',
                    b'{"event":{"completionEnd":{}}}',
                )
            )

        async def await_output(self) -> tuple[None, FakeStream]:
            self.await_output_calls += 1
            return None, self

        async def receive(self) -> object:
            try:
                payload = next(self.payloads)
            except StopIteration as error:
                raise StopAsyncIteration from error
            return SimpleNamespace(value=SimpleNamespace(bytes_=payload))

    transport = AwsNovaSonicTransport(config)
    stream = FakeStream()
    transport._stream = cast(Any, stream)
    transport._state = NovaSessionState.ACTIVE

    events = [event async for event in transport.events()]

    assert events == [
        OutputAudio(pcm16le_24khz=b"\x00\x00"),
        CompletionEnded(),
        OutputAudio(pcm16le_24khz=b"\x00\x00"),
        CompletionEnded(),
    ]
    assert stream.await_output_calls == 5


@pytest.mark.asyncio
async def test_events_stop_on_empty_sdk_result_after_close(
    config: NovaRuntimeConfig,
) -> None:
    class FakeStream:
        def __init__(self) -> None:
            self.await_output_calls = 0

        async def await_output(self) -> tuple[None, FakeStream]:
            self.await_output_calls += 1
            if self.await_output_calls > 1:
                raise AssertionError("transport spun on an empty closed stream")
            return None, self

        async def receive(self) -> object:
            return SimpleNamespace(value=None)

    transport = AwsNovaSonicTransport(config)
    stream = FakeStream()
    transport._stream = cast(Any, stream)
    transport._state = NovaSessionState.CLOSED

    events = [event async for event in transport.events()]

    assert events == []
    assert stream.await_output_calls == 1
