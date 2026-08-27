"""Unit tests for FastAPI Twilio and public-demo endpoints."""

from __future__ import annotations

import asyncio
import base64
import re
import threading
from collections.abc import AsyncIterator, Sequence
from datetime import UTC, datetime

import pytest
import structlog
from fastapi.testclient import TestClient
from httpx import Response
from starlette.websockets import WebSocketDisconnect
from twilio.request_validator import RequestValidator  # type: ignore[import-untyped]

from realtime_voice_agent.audio.codecs import encode_twilio_mulaw_payload
from realtime_voice_agent.config import AppSettings
from realtime_voice_agent.main import _cancel_tasks, create_app
from realtime_voice_agent.nova.events import (
    CompletionEnded,
    InterruptionStarted,
    NovaServerEvent,
    NovaSessionState,
    OutputAudio,
)
from realtime_voice_agent.observability.models import MetricDatum
from realtime_voice_agent.persistence.models import (
    Persona,
    SessionStart,
    SessionTerminal,
    TranscriptTurn,
    TranscriptView,
)
from realtime_voice_agent.readiness import ReadinessErrorCode, ReadinessResult
from realtime_voice_agent.transcript import FinalTranscript, TranscriptSpeaker

_TEST_AUTH_TOKEN = "-".join(("test", "placeholder"))
_TEST_ACCOUNT_SID = "test-account-sid"
_PUBLIC_MEDIA_URL = "wss://example.ngrok.app/media"


class FakeReadiness:
    def __init__(self, result: ReadinessResult) -> None:
        self.result = result
        self.calls = 0

    async def check(self) -> ReadinessResult:
        self.calls += 1
        return self.result


class FakeTelemetry:
    def __init__(self) -> None:
        self.started = False
        self.closed = False
        self.logs: list[dict[str, object]] = []
        self.metrics: list[MetricDatum] = []

    async def start(self) -> None:
        self.started = True

    async def close(self) -> None:
        self.closed = True

    def publish_log(self, event: dict[str, object]) -> bool:
        self.logs.append(event)
        return True

    def publish_metric(self, metric: MetricDatum) -> bool:
        self.metrics.append(metric)
        return True


def test_health_returns_ok_for_valid_local_twilio_config() -> None:
    with TestClient(create_app(_settings())) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "environment": "test"}


def test_health_does_not_invoke_aws_readiness() -> None:
    readiness = FakeReadiness(
        ReadinessResult.unavailable(ReadinessErrorCode.AWS_IDENTITY_UNAVAILABLE)
    )
    with TestClient(create_app(_settings(), readiness_checker=readiness)) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert readiness.calls == 0


def test_ready_returns_safe_200_when_dependencies_are_ready() -> None:
    readiness = FakeReadiness(ReadinessResult.ready())
    with TestClient(create_app(_settings(), readiness_checker=readiness)) as client:
        response = client.get("/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "ready"}
    assert readiness.calls == 1


def test_ready_returns_safe_503_without_sensitive_dependency_details() -> None:
    readiness = FakeReadiness(ReadinessResult.unavailable(ReadinessErrorCode.TABLE_MISSING))
    with TestClient(create_app(_settings(), readiness_checker=readiness)) as client:
        response = client.get("/ready")

    assert response.status_code == 503
    assert response.json() == {
        "status": "not_ready",
        "error_code": "TABLE_MISSING",
    }
    assert set(response.json()) == {"status", "error_code"}


def test_cloudwatch_telemetry_follows_application_lifespan() -> None:
    telemetry = FakeTelemetry()
    app = create_app(
        _settings(cloudwatch_enabled=True),
        telemetry_factory=lambda _config: telemetry,
    )

    with TestClient(app) as client:
        assert telemetry.started is True
        assert client.get("/health").status_code == 200

    assert telemetry.closed is True


def test_voice_returns_connect_stream_twiml_with_valid_signature() -> None:
    fake_credential = "test-placeholder"
    settings = _settings(
        twilio_validate_signatures=True,
        **{"twilio_auth_" + "token": fake_credential},
    )
    form = {"CallSid": "CA00000000000000000000000000000000"}
    signature = RequestValidator(fake_credential).compute_signature(  # type: ignore[no-untyped-call]
        "https://example.ngrok.app/voice",
        form,
    )

    with TestClient(create_app(settings)) as client:
        response = client.post("/voice", data=form, headers={"X-Twilio-Signature": signature})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/xml")
    assert '<Connect><Stream url="wss://example.ngrok.app/media"' in response.text


def test_voice_rejects_invalid_signature_without_echoing_token() -> None:
    fake_credential = "test-placeholder"
    settings = _settings(
        twilio_validate_signatures=True,
        **{"twilio_auth_" + "token": fake_credential},
    )

    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/voice",
            data={"CallSid": "CA00000000000000000000000000000000"},
            headers={"X-Twilio-Signature": "invalid"},
        )

    assert response.status_code == 403
    assert fake_credential not in response.text


def test_demo_voice_menu_speaks_disclaimer_and_configured_persona_names() -> None:
    store = _demo_store()
    call_form = {
        "CallSid": "CA11111111111111111111111111111111",
        "From": "+15555550100",
    }

    with TestClient(
        create_app(
            _settings(demo_mode_enabled=True),
            persistence_store=store,
        )
    ) as client:
        response = _signed_post(client, "/voice", call_form)

    assert response.status_code == 200
    assert "portfolio demonstration" in response.text
    assert "sensitive personal, medical, financial, authentication, or account information" in (
        response.text
    )
    assert "limited to 5 minutes" in response.text
    assert "Care Coordinator" in response.text
    assert "Financial Services Assistant" in response.text
    assert "Travel Concierge" in response.text
    assert "History Guide" in response.text
    assert "/select-persona" in response.text
    assert "<Connect>" not in response.text
    assert call_form["From"] not in response.text


def test_demo_selection_speaks_persona_greeting_before_connecting_stream() -> None:
    store = _demo_store()
    call_form = {
        "CallSid": "CA11111111111111111111111111111114",
        "From": "+15555550108",
    }
    with TestClient(
        create_app(
            _settings(demo_mode_enabled=True),
            persistence_store=store,
        )
    ) as client:
        assert _signed_post(client, "/voice", call_form).status_code == 200
        response = _signed_post(
            client,
            "/select-persona",
            {**call_form, "Digits": "4"},
        )

    greeting = "Hello. Your History Guide is ready. How can I help you today?"
    assert response.status_code == 200
    assert greeting in response.text
    assert response.text.index(greeting) < response.text.index("<Connect>")
    assert store.personas["history-guide"].system_prompt not in response.text
    assert call_form["From"] not in response.text


def test_demo_selection_rejects_invalid_signature() -> None:
    call_form = {
        "CallSid": "CA11111111111111111111111111111112",
        "From": "+15555550106",
    }
    with TestClient(
        create_app(
            _settings(demo_mode_enabled=True),
            persistence_store=_demo_store(),
        )
    ) as client:
        assert _signed_post(client, "/voice", call_form).status_code == 200
        rejected = client.post(
            "/select-persona",
            data={**call_form, "Digits": "1"},
            headers={"X-Twilio-Signature": "invalid"},
        )

    assert rejected.status_code == 403
    assert _TEST_AUTH_TOKEN not in rejected.text


def test_demo_invalid_selection_repeats_menu_without_reserving_media() -> None:
    call_form = {
        "CallSid": "CA11111111111111111111111111111113",
        "From": "+15555550107",
    }
    with TestClient(
        create_app(
            _settings(demo_mode_enabled=True),
            persistence_store=_demo_store(),
        )
    ) as client:
        assert _signed_post(client, "/voice", call_form).status_code == 200
        response = _signed_post(
            client,
            "/select-persona",
            {**call_form, "Digits": "9"},
        )

    assert response.status_code == 200
    assert "selection was not recognized" in response.text
    assert "Press 1 for Care Coordinator" in response.text
    assert "portfolio demonstration" not in response.text
    assert "<Connect>" not in response.text


def test_demo_selection_loads_versioned_persona_in_shared_runtime_without_transcript_text() -> None:
    store = _demo_store()
    selected = store.personas["travel-concierge"]
    nova = FakeNovaTransport(
        output_events=(
            OutputAudio(pcm16le_24khz=b"\x01\x00" * 480),
            FinalTranscript(
                speaker=TranscriptSpeaker.CALLER,
                text="content that must not be retained",
                source_event_id="caller-final",
            ),
            CompletionEnded(),
        )
    )
    call_form = {
        "CallSid": "CA22222222222222222222222222222222",
        "From": "+15555550101",
    }

    with TestClient(
        create_app(
            _settings(demo_mode_enabled=True),
            nova_transport_factory=lambda _config: nova,
            persistence_store=store,
        )
    ) as client:
        reservation_token = _reserve_demo_call(client, call_form=call_form, digit="3")
        with client.websocket_connect("/media", headers=_valid_media_headers()) as websocket:
            websocket.send_json({"event": "connected", "protocol": "Call", "version": "1.0.0"})
            websocket.send_json(
                _start_message(
                    custom_parameters={"demoReservation": reservation_token},
                )
            )
            websocket.send_json(
                {
                    "event": "media",
                    "sequenceNumber": "2",
                    "media": {
                        "chunk": "1",
                        "timestamp": "0",
                        "payload": base64.b64encode(b"\xff" * 160).decode(),
                    },
                }
            )
            assert websocket.receive_json()["event"] == "media"
            websocket.send_json({"event": "stop", "sequenceNumber": "3"})
            assert websocket.receive_json()["event"] == "mark"
            with pytest.raises(WebSocketDisconnect):
                websocket.receive_json()

    assert nova.system_prompt == selected.system_prompt
    assert store.active_persona_reads == 0
    assert store.sessions[0].persona.persona_id == selected.persona_id
    assert store.sessions[0].persona.version == selected.version
    assert store.turns == []
    assert store.terminal_written.wait(timeout=1)
    assert store.terminals[-1].caller_turns == 1
    assert call_form["From"] not in repr(store.sessions)


@pytest.mark.parametrize(
    "limit_overrides",
    [
        {"demo_global_concurrency_limit": 1, "demo_budget_max_calls": 10},
        {"demo_global_concurrency_limit": 2, "demo_budget_max_calls": 1},
    ],
)
def test_demo_capacity_and_budget_rejections_are_spoken_and_hang_up(
    limit_overrides: dict[str, int],
) -> None:
    settings = _settings(demo_mode_enabled=True, **limit_overrides)
    with TestClient(create_app(settings, persistence_store=_demo_store())) as client:
        _reserve_demo_call(
            client,
            call_form={
                "CallSid": "CA33333333333333333333333333333333",
                "From": "+15555550102",
            },
            digit="1",
        )
        rejected = _signed_post(
            client,
            "/voice",
            {
                "CallSid": "CA44444444444444444444444444444444",
                "From": "+15555550103",
            },
        )

    assert rejected.status_code == 200
    assert "portfolio demo is currently unavailable" in rejected.text
    assert "<Hangup" in rejected.text
    assert "budget" not in rejected.text.lower()
    assert "capacity" not in rejected.text.lower()


def test_demo_rate_limit_is_spoken_without_retaining_caller_number(
    capsys: pytest.CaptureFixture[str],
) -> None:
    caller = "+15555550104"
    with TestClient(
        create_app(
            _settings(
                demo_mode_enabled=True,
                demo_rate_limit_max_calls=1,
            ),
            persistence_store=_demo_store(),
        )
    ) as client:
        first = _signed_post(
            client,
            "/voice",
            {"CallSid": "CA55555555555555555555555555555555", "From": caller},
        )
        rejected = _signed_post(
            client,
            "/voice",
            {"CallSid": "CA66666666666666666666666666666666", "From": caller},
        )

    assert "Press 1 for Care Coordinator" in first.text
    assert "per-caller usage limit" in rejected.text
    assert "<Hangup" in rejected.text
    assert caller not in rejected.text
    assert caller not in capsys.readouterr().out


def test_demo_media_requires_reservation_before_persona_or_nova_access() -> None:
    store = _demo_store()
    transport_calls = 0

    def new_transport(_config: object) -> FakeNovaTransport:
        nonlocal transport_calls
        transport_calls += 1
        return FakeNovaTransport()

    with TestClient(
        create_app(
            _settings(demo_mode_enabled=True),
            persistence_store=store,
            nova_transport_factory=new_transport,
        )
    ) as client:
        with client.websocket_connect("/media", headers=_valid_media_headers()) as websocket:
            websocket.send_json({"event": "connected", "protocol": "Call", "version": "1.0.0"})
            websocket.send_json(_start_message())
            with pytest.raises(WebSocketDisconnect) as closed:
                websocket.receive_json()

    assert closed.value.code == 1008
    assert store.active_persona_reads == 0
    assert store.sessions == []
    assert transport_calls == 0


def test_demo_duration_guard_closes_call_with_bounded_nonfailure_reason() -> None:
    store = _demo_store()
    call_form = {
        "CallSid": "CA77777777777777777777777777777777",
        "From": "+15555550105",
    }
    with TestClient(
        create_app(
            _settings(
                demo_mode_enabled=True,
                demo_max_call_duration_seconds=0.01,
            ),
            persistence_store=store,
            nova_transport_factory=lambda _config: FakeNovaTransport(),
        )
    ) as client:
        token = _reserve_demo_call(client, call_form=call_form, digit="4")
        with client.websocket_connect("/media", headers=_valid_media_headers()) as websocket:
            websocket.send_json({"event": "connected", "protocol": "Call", "version": "1.0.0"})
            websocket.send_json(_start_message(custom_parameters={"demoReservation": token}))
            with pytest.raises(WebSocketDisconnect) as closed:
                websocket.receive_json()

    assert closed.value.code == 1000
    assert store.terminal_written.wait(timeout=1)
    assert store.terminals[-1].termination_reason == "DEMO_TIME_LIMIT"
    assert store.terminals[-1].outcome == "DISCONNECTED"


def test_public_api_has_no_outbound_call_route() -> None:
    with TestClient(create_app(_settings())) as client:
        paths = client.get("/openapi.json").json()["paths"]

    assert set(paths) == {"/health", "/ready", "/voice", "/select-persona"}


@pytest.mark.parametrize(
    "signature",
    [None, "malformed-signature"],
)
def test_media_websocket_rejects_unauthenticated_handshake_before_backend_access(
    signature: str | None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    store = FakePersistenceStore()
    transport_calls = 0

    def new_transport(_config: object) -> FakeNovaTransport:
        nonlocal transport_calls
        transport_calls += 1
        return FakeNovaTransport()

    headers = {} if signature is None else {"x-twilio-signature": signature}
    with TestClient(
        create_app(
            _settings(),
            nova_transport_factory=new_transport,
            persistence_store=store,
        )
    ) as client:
        with pytest.raises(WebSocketDisconnect) as denied:
            with client.websocket_connect("/media", headers=headers):
                pytest.fail("Unauthenticated WebSocket was accepted")

    assert denied.value.code == 1008
    assert store.active_persona_reads == 0
    assert transport_calls == 0
    output = capsys.readouterr().out
    assert "twilio_websocket_signature_rejected" in output
    if signature is not None:
        assert signature not in output
    assert _TEST_AUTH_TOKEN not in output


def test_media_websocket_rejects_signature_for_noncanonical_url() -> None:
    signature = RequestValidator(_TEST_AUTH_TOKEN).compute_signature(
        "wss://internal.local/media",
        {},
    )
    store = FakePersistenceStore()

    with TestClient(create_app(_settings(), persistence_store=store)) as client:
        with pytest.raises(WebSocketDisconnect) as denied:
            with client.websocket_connect(
                "/media",
                headers={"x-twilio-signature": signature},
            ):
                pytest.fail("Signature for noncanonical URL was accepted")

    assert denied.value.code == 1008
    assert store.active_persona_reads == 0


def test_media_websocket_bridges_caller_and_nova_audio_bidirectionally() -> None:
    nova = FakeNovaTransport()
    store = FakePersistenceStore()
    with TestClient(
        create_app(
            _settings(),
            nova_transport_factory=lambda _config: nova,
            persistence_store=store,
        )
    ) as client:
        with client.websocket_connect("/media", headers=_valid_media_headers()) as websocket:
            websocket.send_json({"event": "connected", "protocol": "Call", "version": "1.0.0"})
            websocket.send_json(_start_message())

            websocket.send_json(
                {
                    "event": "media",
                    "sequenceNumber": "2",
                    "media": {
                        "track": "inbound",
                        "chunk": "1",
                        "timestamp": "20",
                        "payload": encode_twilio_mulaw_payload(b"\x00\x00" * 160),
                    },
                }
            )
            assert nova.audio_received.wait(timeout=1)
            assert nova.sent_audio
            model_outbound = websocket.receive_json()
            model_payload = model_outbound["media"]["payload"]
            assert isinstance(model_payload, str)
            model_mulaw = base64.b64decode(model_payload, validate=True)
            assert model_outbound["event"] == "media"
            assert model_outbound["streamSid"] == "MZ00000000000000000000000000000000"
            assert len(model_mulaw) == 160
            assert model_mulaw[:4] != b"RIFF"
            mark = websocket.receive_json()
            assert mark == {
                "event": "mark",
                "streamSid": "MZ00000000000000000000000000000000",
                "mark": {"name": "response-1"},
            }
            websocket.send_json(
                {
                    "event": "mark",
                    "sequenceNumber": "3",
                    "mark": {"name": "response-1"},
                }
            )
            websocket.send_json({"event": "stop", "sequenceNumber": "4", "stop": {}})
            with pytest.raises(WebSocketDisconnect) as closed:
                websocket.receive_json()
            assert closed.value.code == 1000
    assert nova.system_prompt == "Answer as the active test persona."
    assert store.active_persona_reads == 1
    assert store.sessions


def test_media_websocket_clears_interrupted_audio_before_new_generation() -> None:
    nova = FakeNovaTransport(
        output_events=(
            OutputAudio(pcm16le_24khz=b"\x01\x00" * 480),
            InterruptionStarted(source_event_id="interruption-main-1"),
            OutputAudio(pcm16le_24khz=b"\x02\x00" * 480),
            CompletionEnded(),
        )
    )
    with TestClient(
        create_app(
            _settings(),
            nova_transport_factory=lambda _config: nova,
            persistence_store=FakePersistenceStore(),
        )
    ) as client:
        with client.websocket_connect("/media", headers=_valid_media_headers()) as websocket:
            websocket.send_json({"event": "connected", "protocol": "Call", "version": "1.0.0"})
            websocket.send_json(_start_message())
            websocket.send_json(
                {
                    "event": "media",
                    "sequenceNumber": "2",
                    "media": {
                        "track": "inbound",
                        "chunk": "1",
                        "timestamp": "20",
                        "payload": encode_twilio_mulaw_payload(b"\x00\x00" * 160),
                    },
                }
            )

            messages: list[dict[str, object]] = []
            while not messages or messages[-1]["event"] != "mark":
                messages.append(websocket.receive_json())

            clear_index = next(
                index for index, message in enumerate(messages) if message["event"] == "clear"
            )
            assert any(message["event"] == "media" for message in messages[clear_index + 1 :])
            assert messages[-1] == {
                "event": "mark",
                "streamSid": "MZ00000000000000000000000000000000",
                "mark": {"name": "response-1"},
            }
            websocket.send_json({"event": "stop", "sequenceNumber": "3", "stop": {}})
            with pytest.raises(WebSocketDisconnect) as closed:
                websocket.receive_json()
            assert closed.value.code == 1000


def test_five_sequential_websocket_calls_do_not_reuse_transport_or_stream() -> None:
    transports: list[FakeNovaTransport] = []

    def new_transport(_config: object) -> FakeNovaTransport:
        transport = FakeNovaTransport()
        transports.append(transport)
        return transport

    with TestClient(
        create_app(
            _settings(),
            nova_transport_factory=new_transport,
            persistence_store=FakePersistenceStore(),
        )
    ) as client:
        for digit in range(5):
            with client.websocket_connect(
                "/media",
                headers=_valid_media_headers(),
            ) as websocket:
                websocket.send_json(_start_message(stream_digit=str(digit)))
                websocket.send_json(
                    {
                        "event": "media",
                        "sequenceNumber": "2",
                        "media": {
                            "track": "inbound",
                            "chunk": "1",
                            "timestamp": "20",
                            "payload": encode_twilio_mulaw_payload(b"\x00\x00" * 160),
                        },
                    }
                )
                outbound = websocket.receive_json()
                assert outbound["streamSid"] == f"MZ{str(digit) * 32}"
                assert websocket.receive_json()["event"] == "mark"
                websocket.send_json({"event": "stop", "sequenceNumber": "3", "stop": {}})
                with pytest.raises(WebSocketDisconnect) as closed:
                    websocket.receive_json()
                assert closed.value.code == 1000

    assert len({id(transport) for transport in transports}) == 5
    assert all(transport.state is NovaSessionState.CLOSED for transport in transports)


def test_media_websocket_rejects_invalid_start_format() -> None:
    message = _start_message()
    start = message["start"]
    assert isinstance(start, dict)
    media_format = start["mediaFormat"]
    assert isinstance(media_format, dict)
    media_format["sampleRate"] = 16_000

    with TestClient(create_app(_settings(), persistence_store=FakePersistenceStore())) as client:
        with client.websocket_connect(
            "/media",
            headers=_valid_media_headers(),
        ) as websocket:
            websocket.send_json(message)
            try:
                websocket.receive_json()
            except WebSocketDisconnect as error:
                assert error.code == 1003
            else:  # pragma: no cover
                raise AssertionError("WebSocket stayed open after invalid media format")


def test_media_websocket_rejects_mismatched_start_account_before_nova_opens() -> None:
    nova = FakeNovaTransport()
    store = FakePersistenceStore()
    with TestClient(
        create_app(
            _settings(),
            nova_transport_factory=lambda _config: nova,
            persistence_store=store,
        )
    ) as client:
        with client.websocket_connect(
            "/media",
            headers=_valid_media_headers(),
        ) as websocket:
            message = _start_message()
            start = message["start"]
            assert isinstance(start, dict)
            start["accountSid"] = "mismatched-account-sid"
            websocket.send_json(message)
            with pytest.raises(WebSocketDisconnect) as closed:
                websocket.receive_json()

    assert closed.value.code == 1003
    assert nova.system_prompt is None
    assert store.sessions == []


@pytest.mark.asyncio
async def test_bridge_cleanup_does_not_block_on_uncooperative_worker() -> None:
    started = asyncio.Event()
    finished = asyncio.Event()

    async def worker() -> None:
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await asyncio.sleep(0.1)
        finally:
            finished.set()

    task = asyncio.create_task(worker())
    await asyncio.wait_for(started.wait(), timeout=1)

    await asyncio.wait_for(
        _cancel_tasks(
            (task,),
            timeout_seconds=0.01,
            logger=structlog.get_logger(),
        ),
        timeout=0.1,
    )

    assert not task.done()
    await asyncio.wait_for(finished.wait(), timeout=1)


class FakeNovaTransport:
    def __init__(self, *, output_events: tuple[NovaServerEvent, ...] | None = None) -> None:
        self._state = NovaSessionState.NEW
        self.audio_received = threading.Event()
        self._output_ready = asyncio.Event()
        self._output_events = output_events or (
            OutputAudio(pcm16le_24khz=b"\x01\x00" * 480),
            CompletionEnded(),
        )
        self.sent_audio: list[bytes] = []
        self._closed = asyncio.Event()
        self.system_prompt: str | None = None

    @property
    def state(self) -> NovaSessionState:
        return self._state

    async def start(
        self,
        *,
        system_prompt: str,
        history: tuple[object, ...] = (),
    ) -> None:
        self.system_prompt = system_prompt
        assert system_prompt
        self._state = NovaSessionState.ACTIVE

    async def start_audio_input(self) -> None:
        assert self._state is NovaSessionState.ACTIVE

    async def send_audio(self, pcm16le_16khz: bytes) -> None:
        self.sent_audio.append(pcm16le_16khz)
        self.audio_received.set()
        self._output_ready.set()

    async def events(self) -> AsyncIterator[NovaServerEvent]:
        await self._output_ready.wait()
        for event in self._output_events:
            yield event
        await self._closed.wait()

    async def finish_input(self) -> None:
        return None

    async def close(self) -> None:
        self._state = NovaSessionState.CLOSED
        self._output_ready.set()
        self._closed.set()


class FakePersistenceStore:
    def __init__(self) -> None:
        now = datetime(2026, 8, 4, tzinfo=UTC)
        self.active_persona_reads = 0
        self.persona = Persona(
            persona_id="test-persona",
            name="Test persona",
            system_prompt="Answer as the active test persona.",
            voice_id="matthew",
            version=2,
            active=True,
            created_at=now,
            updated_at=now,
        )
        self.personas: dict[str, Persona] = {self.persona.persona_id: self.persona}
        self.sessions: list[SessionStart] = []
        self.turns: list[TranscriptTurn] = []
        self.terminals: list[SessionTerminal] = []
        self.terminal_written = threading.Event()

    def list_personas(self) -> Sequence[Persona]:
        return tuple(self.personas.values())

    def get_persona(self, persona_id: str) -> Persona | None:
        return self.personas.get(persona_id)

    def get_active_persona(self) -> Persona:
        self.active_persona_reads += 1
        return self.persona

    def put_persona(
        self,
        *,
        persona_id: str,
        name: str,
        system_prompt: str,
        voice_id: str,
        expected_version: int,
    ) -> Persona:
        raise NotImplementedError

    def activate_persona(self, persona_id: str, *, expected_version: int) -> Persona:
        raise NotImplementedError

    def create_session(self, session: SessionStart) -> None:
        self.sessions.append(session)

    def mark_session_active(self, session_id: str, activated_at: str) -> None:
        return None

    def append_transcript_turn(self, turn: TranscriptTurn) -> bool:
        self.turns.append(turn)
        return True

    def finish_session(self, terminal: SessionTerminal) -> bool:
        self.terminals.append(terminal)
        self.terminal_written.set()
        return True

    def get_transcript(
        self,
        *,
        session_id: str | None = None,
        call_sid: str | None = None,
    ) -> TranscriptView:
        raise NotImplementedError


def _settings(**overrides: object) -> AppSettings:
    values = {
        "_env_file": None,
        "app_env": "test",
        "public_base_url": "https://example.ngrok.app",
        "twilio_account_sid": _TEST_ACCOUNT_SID,
        "twilio_auth_token": _TEST_AUTH_TOKEN,
        "twilio_validate_signatures": True,
        "aws_region": "us-east-1",
    }
    values.update(overrides)
    return AppSettings(**values)


def _start_message(
    *,
    stream_digit: str = "0",
    custom_parameters: dict[str, str] | None = None,
) -> dict[str, object]:
    start: dict[str, object] = {
        "streamSid": f"MZ{stream_digit * 32}",
        "callSid": f"CA{stream_digit * 32}",
        "accountSid": _TEST_ACCOUNT_SID,
        "mediaFormat": {"encoding": "audio/x-mulaw", "sampleRate": 8000, "channels": 1},
    }
    if custom_parameters is not None:
        start["customParameters"] = custom_parameters
    return {
        "event": "start",
        "sequenceNumber": "1",
        "start": start,
    }


def _demo_store() -> FakePersistenceStore:
    store = FakePersistenceStore()
    now = datetime(2026, 8, 4, tzinfo=UTC)
    definitions = (
        ("care-coordinator", "Care Coordinator", "Coordinate care safely."),
        (
            "financial-services-assistant",
            "Financial Services Assistant",
            "Explain general financial services.",
        ),
        ("travel-concierge", "Travel Concierge", "Plan concise travel options."),
        ("history-guide", "History Guide", "Explain history with evidence."),
    )
    for version, (persona_id, name, prompt) in enumerate(definitions, start=1):
        store.personas[persona_id] = Persona(
            persona_id=persona_id,
            name=name,
            system_prompt=prompt,
            voice_id="matthew",
            version=version,
            active=False,
            created_at=now,
            updated_at=now,
        )
    return store


def _signed_post(
    client: TestClient,
    path: str,
    form: dict[str, str],
) -> Response:
    signature = RequestValidator(_TEST_AUTH_TOKEN).compute_signature(
        f"https://example.ngrok.app{path}",
        form,
    )
    return client.post(
        path,
        data=form,
        headers={"X-Twilio-Signature": signature},
    )


def _reserve_demo_call(
    client: TestClient,
    *,
    call_form: dict[str, str],
    digit: str,
) -> str:
    menu = _signed_post(client, "/voice", call_form)
    assert menu.status_code == 200
    selection_form = {**call_form, "Digits": digit}
    selection = _signed_post(client, "/select-persona", selection_form)
    assert selection.status_code == 200
    match = re.search(
        r'<Parameter name="demoReservation" value="([^"]+)"',
        selection.text,
    )
    assert match is not None
    assert "care-coordinator" not in selection.text
    assert "financial-services-assistant" not in selection.text
    assert "travel-concierge" not in selection.text
    assert "history-guide" not in selection.text
    return match.group(1)


def _valid_media_headers() -> dict[str, str]:
    signature = RequestValidator(_TEST_AUTH_TOKEN).compute_signature(
        _PUBLIC_MEDIA_URL,
        {},
    )
    return {"x-twilio-signature": signature}
