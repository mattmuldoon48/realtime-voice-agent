"""Amazon Bedrock implementation of the Nova streaming boundary."""

from __future__ import annotations

import asyncio
import os
import uuid
from collections.abc import AsyncIterator
from typing import Any

from aws_sdk_bedrock_runtime.client import (  # type: ignore[import-untyped]
    AsyncBedrockRuntimeClient,
)
from aws_sdk_bedrock_runtime.config import Config  # type: ignore[import-untyped]
from aws_sdk_bedrock_runtime.models import (  # type: ignore[import-untyped]
    BidirectionalInputPayloadPart,
    InvokeModelWithBidirectionalStreamInputChunk,
    InvokeModelWithBidirectionalStreamOperationInput,
)
from smithy_aws_core.identity import (
    AWSCredentialsIdentity,
    AWSCredentialsResolver,
    ContainerCredentialsResolver,
    IdentityChain,
)
from smithy_http.aio.crt import AWSCRTHTTPClient
from smithy_http.aio.interfaces import HTTPClient

from realtime_voice_agent.config import NovaRuntimeConfig
from realtime_voice_agent.nova.events import (
    NovaEventIds,
    NovaEventParser,
    NovaProtocolError,
    NovaServerEvent,
    NovaSessionState,
    build_audio_content_start,
    build_audio_input,
    build_content_end,
    build_history_events,
    build_initialization_events,
    build_interactive_text_events,
    build_prompt_end,
    build_session_end,
    transition_state,
)
from realtime_voice_agent.transcript import ConversationHistoryTurn

_ECS_CREDENTIAL_ENV_VARS = (
    "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
    "AWS_CONTAINER_CREDENTIALS_FULL_URI",
)


async def _create_credentials_resolver(
    config: NovaRuntimeConfig,
    http_client: HTTPClient,
) -> AWSCredentialsResolver:
    """Use native ECS task-role credentials when the standard chain cannot register them."""
    if any(os.getenv(name) for name in _ECS_CREDENTIAL_ENV_VARS):
        return ContainerCredentialsResolver(http_client)
    return await IdentityChain.create(
        AWSCredentialsIdentity,
        profile_name=config.profile,
        region_override=config.region,
        http_client=http_client,
    )


class AwsNovaSonicTransport:
    """Contain all experimental AWS SDK types and event-stream operations."""

    def __init__(self, config: NovaRuntimeConfig) -> None:
        self._config = config
        self._state = NovaSessionState.NEW
        self._client: AsyncBedrockRuntimeClient | None = None
        self._stream: Any | None = None
        self._ids = NovaEventIds(
            prompt_name=str(uuid.uuid4()),
            system_content_name=str(uuid.uuid4()),
            audio_content_name=str(uuid.uuid4()),
        )
        self._event_parser = NovaEventParser()
        self._session_initialized = False
        self._audio_started = False
        self._audio_finished = False
        self._close_lock = asyncio.Lock()

    @property
    def state(self) -> NovaSessionState:
        """Return the current transport lifecycle state."""
        return self._state

    async def start(
        self,
        *,
        system_prompt: str,
        history: tuple[ConversationHistoryTurn, ...] = (),
    ) -> None:
        """Open the stream and send initialization plus bounded conversation history."""
        self._state = transition_state(self._state, NovaSessionState.STARTING)
        try:
            http_client = AWSCRTHTTPClient()
            credentials = await _create_credentials_resolver(self._config, http_client)
            sdk_config = Config(
                endpoint_uri=(f"https://bedrock-runtime.{self._config.region}.amazonaws.com"),
                region=self._config.region,
                aws_credentials_identity_resolver=credentials,
                transport=http_client,
            )
            self._client = AsyncBedrockRuntimeClient(config=sdk_config)
            self._stream = await self._client.invoke_model_with_bidirectional_stream(
                InvokeModelWithBidirectionalStreamOperationInput(model_id=self._config.model_id)
            )
            for event in build_initialization_events(
                ids=self._ids,
                config=self._config,
                system_prompt=system_prompt,
            ):
                await self._send(event)
            for event in build_history_events(
                prompt_name=self._ids.prompt_name,
                history=history,
            ):
                await self._send(event)
            self._session_initialized = True
            self._state = transition_state(self._state, NovaSessionState.ACTIVE)
        except Exception:
            self._state = transition_state(self._state, NovaSessionState.FAILED)
            await self._close_input_stream()
            raise

    async def send_text(self, text: str) -> None:
        """Send one interactive USER text turn before opening caller audio."""
        self._require_state(NovaSessionState.ACTIVE)
        if self._audio_started:
            raise NovaProtocolError("Nova interactive text must precede audio input")
        for event in build_interactive_text_events(
            prompt_name=self._ids.prompt_name,
            text=text,
        ):
            await self._send(event)

    async def start_audio_input(self) -> None:
        """Declare the PCM16LE mono 16 kHz input content block once."""
        self._require_state(NovaSessionState.ACTIVE)
        if self._audio_started:
            raise NovaProtocolError("Nova audio input has already started")
        await self._send(build_audio_content_start(ids=self._ids, config=self._config))
        self._audio_started = True

    async def send_audio(self, pcm16le_16khz: bytes) -> None:
        """Send one validated raw PCM16LE mono 16 kHz chunk."""
        self._require_state(NovaSessionState.ACTIVE)
        if not self._audio_started or self._audio_finished:
            raise NovaProtocolError("Nova audio input is not open")
        await self._send(build_audio_input(ids=self._ids, pcm16le_16khz=pcm16le_16khz))

    async def events(self) -> AsyncIterator[NovaServerEvent]:
        """Yield stable domain events while hiding AWS event-stream types."""
        stream = self._stream
        if stream is None:
            raise NovaProtocolError("Nova stream is not open")

        while True:
            try:
                output = await stream.await_output()
                result = await output[1].receive()
            except StopAsyncIteration:
                return

            value = getattr(result, "value", None)
            payload = getattr(value, "bytes_", None)
            if not isinstance(payload, bytes) or not payload:
                if self._state is NovaSessionState.CLOSED:
                    return
                continue
            event = self._event_parser.parse(payload)
            if event is not None:
                yield event

    async def finish_input(self) -> None:
        """End the input content block once."""
        if not self._audio_started or self._audio_finished:
            return
        if self._state not in {NovaSessionState.ACTIVE, NovaSessionState.CLOSING}:
            return
        await self._send(
            build_content_end(
                prompt_name=self._ids.prompt_name,
                content_name=self._ids.audio_content_name,
            )
        )
        self._audio_finished = True

    async def close(self) -> None:
        """Idempotently end input, prompt, session, and the SDK input stream."""
        async with self._close_lock:
            if self._state is NovaSessionState.CLOSED:
                return
            if self._state is NovaSessionState.NEW:
                self._state = transition_state(self._state, NovaSessionState.CLOSED)
                return
            if self._state is not NovaSessionState.CLOSING:
                self._state = transition_state(self._state, NovaSessionState.CLOSING)

            cleanup_error: Exception | None = None
            try:
                await self.finish_input()
                if self._session_initialized:
                    await self._send(build_prompt_end(prompt_name=self._ids.prompt_name))
                    await self._send(build_session_end())
            except Exception as error:
                cleanup_error = error
            try:
                await self._close_input_stream()
            except Exception as error:
                cleanup_error = cleanup_error or error

            if cleanup_error is None:
                self._state = transition_state(self._state, NovaSessionState.CLOSED)
                return

            self._state = transition_state(self._state, NovaSessionState.FAILED)
            self._state = transition_state(self._state, NovaSessionState.CLOSED)
            raise cleanup_error

    async def _send(self, payload: bytes) -> None:
        if self._stream is None:
            raise NovaProtocolError("Nova stream is not open")
        event = InvokeModelWithBidirectionalStreamInputChunk(
            value=BidirectionalInputPayloadPart(bytes_=payload)
        )
        await self._stream.input_stream.send(event)

    async def _close_input_stream(self) -> None:
        if self._stream is None:
            return
        await self._stream.input_stream.close()
        self._stream = None

    def _require_state(self, expected: NovaSessionState) -> None:
        if self._state is not expected:
            raise NovaProtocolError(
                f"Nova transport must be {expected}; current state is {self._state}"
            )
