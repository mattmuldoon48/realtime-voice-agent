"""FastAPI entrypoint for Twilio Media Streams and Amazon Nova 2 Sonic."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import replace
from typing import cast

from fastapi import FastAPI, HTTPException, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from starlette.datastructures import FormData
from structlog.typing import FilteringBoundLogger

from realtime_voice_agent.config import (
    AppSettings,
    ConfigurationError,
    DemoRuntimeConfig,
    NovaRuntimeConfig,
    ObservabilityRuntimeConfig,
    PersistenceRuntimeConfig,
    ReadinessRuntimeConfig,
    TwilioRuntimeConfig,
)
from realtime_voice_agent.demo import (
    DemoAdmissionController,
    DemoCallLease,
    DemoRejectionReason,
    keyed_demo_identifier,
)
from realtime_voice_agent.nova.aws_transport import AwsNovaSonicTransport
from realtime_voice_agent.nova.continuation import ContinuingNovaSonicTransport
from realtime_voice_agent.nova.transport import NovaSonicTransport
from realtime_voice_agent.observability.cloudwatch import (
    TelemetryService,
    create_cloudwatch_telemetry,
)
from realtime_voice_agent.observability.logging import (
    configure_local_logging,
    emit_local_event,
    get_logger,
)
from realtime_voice_agent.observability.models import NullTelemetryPublisher
from realtime_voice_agent.persistence.dynamodb import DynamoPersistenceStore
from realtime_voice_agent.persistence.errors import PersistenceError
from realtime_voice_agent.persistence.ports import PersistenceStore
from realtime_voice_agent.readiness import (
    ReadinessChecker,
    create_readiness_service,
)
from realtime_voice_agent.telephony.events import (
    ConnectedEvent,
    StartEvent,
    TwilioProtocolError,
    parse_twilio_event,
)
from realtime_voice_agent.telephony.session import (
    CallSession,
    CallSessionError,
    CallTerminationReason,
)
from realtime_voice_agent.telephony.webhook import (
    TwilioSignatureError,
    build_connect_stream_twiml,
    build_demo_menu_twiml,
    build_demo_rejection_twiml,
    validate_twilio_signature,
    validate_twilio_websocket_signature,
)

type NovaTransportFactory = Callable[[NovaRuntimeConfig], NovaSonicTransport]
type TelemetryFactory = Callable[[ObservabilityRuntimeConfig], TelemetryService]
type ReadinessFactory = Callable[
    [ReadinessRuntimeConfig, PersistenceRuntimeConfig],
    ReadinessChecker,
]


def create_app(
    settings: AppSettings | None = None,
    *,
    nova_transport_factory: NovaTransportFactory | None = None,
    persistence_store: PersistenceStore | None = None,
    telemetry_factory: TelemetryFactory | None = None,
    readiness_checker: ReadinessChecker | None = None,
    readiness_factory: ReadinessFactory | None = None,
    demo_admission_controller: DemoAdmissionController | None = None,
) -> FastAPI:
    """Create the local FastAPI app with validated configuration in lifespan."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        try:
            loaded_settings = settings or AppSettings()
            nova_config = loaded_settings.to_runtime_config()
            config = loaded_settings.to_twilio_runtime_config()
            persistence_config = loaded_settings.to_persistence_runtime_config()
            readiness_config = loaded_settings.to_readiness_runtime_config()
            observability_config = loaded_settings.to_observability_runtime_config()
            demo_config = loaded_settings.to_demo_runtime_config()
            runtime_store = persistence_store or DynamoPersistenceStore(persistence_config)
        except (ConfigurationError, PersistenceError, ValidationError) as error:
            configure_local_logging(level="INFO")
            get_logger().error(
                "startup_configuration_error",
                message="Application configuration failed",
                error_type=type(error).__name__,
            )
            raise
        runtime_telemetry: TelemetryService = NullTelemetryPublisher()
        if observability_config.enabled:
            try:
                runtime_telemetry = (telemetry_factory or create_cloudwatch_telemetry)(
                    observability_config
                )
                await runtime_telemetry.start()
            except Exception as error:
                configure_local_logging(level=config.log_level, service=config.service_name)
                get_logger(service=config.service_name).bind(environment=config.app_env).error(
                    "cloudwatch_telemetry_start_failed",
                    error_type=type(error).__name__,
                )
                runtime_telemetry = NullTelemetryPublisher()
        configure_local_logging(
            level=config.log_level,
            service=config.service_name,
            cloud_log_sink=(
                runtime_telemetry.publish_log if observability_config.enabled else None
            ),
        )
        app.state.config = config
        app.state.nova_config = nova_config
        app.state.nova_transport_factory = nova_transport_factory or AwsNovaSonicTransport
        app.state.persistence_config = persistence_config
        app.state.persistence_store = runtime_store
        app.state.readiness_config = readiness_config
        app.state.readiness_checker = readiness_checker
        app.state.readiness_factory = readiness_factory or create_readiness_service
        app.state.observability_config = observability_config
        app.state.telemetry = runtime_telemetry
        app.state.demo_config = demo_config
        app.state.demo_admission = demo_admission_controller or DemoAdmissionController(demo_config)
        app.state.logger = get_logger(service=config.service_name).bind(environment=config.app_env)
        try:
            yield
        finally:
            await runtime_telemetry.close()
            configure_local_logging(level=config.log_level, service=config.service_name)

    app = FastAPI(title="Real-Time Voice Agent", lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict[str, str]:
        config = _app_config(app)
        return {"status": "ok", "environment": config.app_env}

    @app.get("/ready")
    async def ready() -> JSONResponse:
        checker = await _app_readiness_checker(app)
        result = await checker.check()
        if not result.is_ready:
            _app_logger(app).info(
                "readiness_check_failed",
                error_code=(
                    result.error_code.value
                    if result.error_code is not None
                    else "READINESS_CHECK_FAILED"
                ),
            )
        return JSONResponse(
            status_code=200 if result.is_ready else 503,
            content=result.to_document(),
        )

    @app.post("/voice")
    async def voice(request: Request) -> Response:
        config = _app_config(app)
        logger = _app_logger(app)
        form_values = _single_value_form(await request.form())
        if config.validate_signatures:
            try:
                validate_twilio_signature(
                    auth_token=_required_auth_token(config),
                    public_url=f"{config.public_base_url}/voice",
                    form=form_values,
                    signature=request.headers.get("X-Twilio-Signature"),
                )
            except TwilioSignatureError as error:
                logger.info("twilio_signature_rejected", reason=type(error).__name__)
                raise HTTPException(status_code=403, detail="Invalid Twilio signature") from error
        demo_config = _app_demo_config(app)
        if not demo_config.enabled:
            twiml = build_connect_stream_twiml(media_ws_url=config.public_media_ws_url)
            logger.info("twilio_voice_webhook_accepted")
            return Response(content=twiml, media_type="application/xml")

        call_sid = form_values.get("CallSid")
        if not call_sid:
            return _demo_rejection_response()
        secret = _demo_identifier_secret(config)
        rejection = await _app_demo_admission(app).admit_entry(
            caller_key=keyed_demo_identifier(
                value=form_values.get("From"),
                secret=secret,
                domain="caller",
            ),
            call_key=keyed_demo_identifier(value=call_sid, secret=secret, domain="call"),
        )
        if rejection is not None:
            logger.info("demo_call_entry_rejected", reason=rejection.value)
            return _demo_rejection_response(
                rate_limited=rejection is DemoRejectionReason.RATE_LIMIT
            )
        try:
            persona_options = await _load_demo_persona_options(app)
        except PersistenceError as error:
            logger.error("demo_persona_menu_unavailable", error_code=str(error))
            return _demo_rejection_response()
        twiml = build_demo_menu_twiml(
            selection_url=f"{config.public_base_url}/select-persona",
            persona_options=persona_options,
            max_duration_seconds=demo_config.max_call_duration_seconds,
        )
        logger.info("demo_voice_menu_presented")
        return Response(content=twiml, media_type="application/xml")

    @app.post("/select-persona")
    async def select_persona(request: Request) -> Response:
        config = _app_config(app)
        demo_config = _app_demo_config(app)
        if not demo_config.enabled:
            raise HTTPException(status_code=404, detail="Not found")
        form_values = _single_value_form(await request.form())
        if config.validate_signatures:
            try:
                validate_twilio_signature(
                    auth_token=_required_auth_token(config),
                    public_url=f"{config.public_base_url}/select-persona",
                    form=form_values,
                    signature=request.headers.get("X-Twilio-Signature"),
                )
            except TwilioSignatureError as error:
                _app_logger(app).info(
                    "twilio_signature_rejected",
                    reason=type(error).__name__,
                )
                raise HTTPException(status_code=403, detail="Invalid Twilio signature") from error
        choice = next(
            (
                configured
                for configured in demo_config.persona_choices
                if configured.digit == form_values.get("Digits")
            ),
            None,
        )
        if choice is None:
            try:
                persona_options = await _load_demo_persona_options(app)
            except PersistenceError:
                return _demo_rejection_response()
            twiml = build_demo_menu_twiml(
                selection_url=f"{config.public_base_url}/select-persona",
                persona_options=persona_options,
                max_duration_seconds=demo_config.max_call_duration_seconds,
                invalid_selection=True,
            )
            return Response(content=twiml, media_type="application/xml")
        call_sid = form_values.get("CallSid")
        if not call_sid:
            return _demo_rejection_response()
        try:
            persona = await asyncio.to_thread(
                _app_persistence_store(app).get_persona,
                choice.persona_id,
            )
        except PersistenceError as error:
            _app_logger(app).error(
                "demo_selected_persona_unavailable",
                error_code=str(error),
            )
            return _demo_rejection_response()
        if persona is None:
            _app_logger(app).error(
                "demo_selected_persona_unavailable",
                persona_id=choice.persona_id,
            )
            return _demo_rejection_response()
        secret = _demo_identifier_secret(config)
        reservation = await _app_demo_admission(app).reserve(
            call_key=keyed_demo_identifier(value=call_sid, secret=secret, domain="call"),
            persona_id=persona.persona_id,
            persona_version=persona.version,
        )
        if isinstance(reservation, DemoRejectionReason):
            _app_logger(app).info(
                "demo_persona_selection_rejected",
                reason=reservation.value,
            )
            return _demo_rejection_response()
        twiml = build_connect_stream_twiml(
            media_ws_url=config.public_media_ws_url,
            custom_parameters={"demoReservation": reservation.token},
        )
        _app_logger(app).info(
            "demo_persona_selected",
            persona_id=persona.persona_id,
            persona_version=persona.version,
        )
        return Response(content=twiml, media_type="application/xml")

    @app.websocket("/media")
    async def media(websocket: WebSocket) -> None:
        config = _app_config(app)
        if config.validate_signatures:
            try:
                validate_twilio_websocket_signature(
                    auth_token=_required_auth_token(config),
                    public_url=config.public_media_ws_url,
                    signature=websocket.headers.get("x-twilio-signature"),
                )
            except TwilioSignatureError:
                emit_local_event(
                    "twilio_websocket_signature_rejected",
                    level="warning",
                    service=config.service_name,
                    environment=config.app_env,
                    error_code="TWILIO_WEBSOCKET_SIGNATURE_INVALID",
                )
                await websocket.close(code=1008)
                return
        await websocket.accept()
        demo_config = _app_demo_config(app)
        admission = _app_demo_admission(app)
        lease: DemoCallLease | None = None
        startup_events: tuple[ConnectedEvent | StartEvent, ...] = ()
        if demo_config.enabled:
            try:
                connected, started, lease = await _receive_demo_startup(
                    websocket=websocket,
                    config=config,
                    demo_config=demo_config,
                    admission=admission,
                )
            except WebSocketDisconnect:
                return
            except (TimeoutError, json.JSONDecodeError, TwilioProtocolError) as error:
                _app_logger(app).info(
                    "demo_media_startup_rejected",
                    error_code=getattr(error, "code", "DEMO_MEDIA_STARTUP_INVALID"),
                )
                await websocket.close(code=1008)
                return
            startup_events = (connected, started)

        persistence_config = _app_persistence_config(app)
        store = _app_persistence_store(app)
        try:
            if lease is None:
                persona = await asyncio.to_thread(store.get_active_persona)
            else:
                selected_persona = await asyncio.to_thread(store.get_persona, lease.persona_id)
                if selected_persona is None or selected_persona.version != lease.persona_version:
                    raise PersistenceError("DEMO_PERSONA_VERSION_UNAVAILABLE")
                persona = selected_persona
        except PersistenceError as error:
            _app_logger(app).error(
                "call_persona_load_failed",
                error_code=str(error),
            )
            if lease is not None:
                await admission.release(lease)
            await websocket.close(code=1011)
            return

        nova_config = replace(_app_nova_config(app), voice_id=persona.voice_id)
        session = CallSession(
            logger=_app_logger(app),
            expected_twilio_account_sid=config.twilio_account_sid,
            malformed_frame_limit=config.malformed_media_frame_limit,
            audio_queue_max_frames=config.twilio_audio_queue_max_frames,
            outbound_queue_max_frames=config.twilio_outbound_queue_max_frames,
            nova=ContinuingNovaSonicTransport(
                factory=lambda: _nova_transport_factory(app)(nova_config),
                rotation_seconds=nova_config.session_rotation_seconds,
            ),
            persona=persona.snapshot(),
            model_id=nova_config.model_id,
            session_repository=store,
            persistence_queue_max_events=persistence_config.queue_max_events,
            transcript_retention_days=persistence_config.transcript_retention_days,
            persist_transcripts=(demo_config.persist_transcripts if demo_config.enabled else True),
            initial_text_prompt="Hello" if demo_config.enabled else None,
            cleanup_timeout_seconds=persistence_config.cleanup_timeout_seconds,
            persistence_max_attempts=persistence_config.max_attempts,
            persistence_retry_base_delay_seconds=persistence_config.retry_base_delay_seconds,
            telemetry=_app_telemetry(app),
            environment=config.app_env,
        )
        tasks: tuple[asyncio.Task[None], ...] = ()
        try:
            for event in startup_events:
                session.handle_event(event)
            receiver = asyncio.create_task(
                _receive_twilio_media(websocket=websocket, session=session),
                name="twilio-media-receiver",
            )
            sender = asyncio.create_task(
                _send_twilio_media(websocket=websocket, session=session),
                name="twilio-media-sender",
            )
            failure_watcher = asyncio.create_task(
                session.wait_for_failure(),
                name="call-session-failure-watcher",
            )
            tasks = (receiver, sender, failure_watcher)
            if demo_config.enabled:
                tasks += (
                    asyncio.create_task(
                        _enforce_demo_duration(
                            session=session,
                            duration_seconds=demo_config.max_call_duration_seconds,
                        ),
                        name="demo-call-duration-guard",
                    ),
                )
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                task.result()
            if session.closed:
                await _cancel_tasks(
                    tasks,
                    timeout_seconds=persistence_config.cleanup_timeout_seconds,
                    logger=_app_logger(app),
                )
                tasks = ()
                await _close_call_session(session)
                await websocket.close(code=1000)
        except WebSocketDisconnect:
            _app_logger(app).info("twilio_websocket_disconnected")
            session.disconnect()
        except CallSessionError as error:
            _app_logger(app).error("call_session_failed", error_code=error.code)
            session.fail(error.code)
            await websocket.close(code=1011)
        except (json.JSONDecodeError, TwilioProtocolError) as error:
            error_code = getattr(error, "code", "TWILIO_INVALID_JSON")
            session.fail(error_code, CallTerminationReason.PROTOCOL_ERROR)
            _app_logger(app).info(
                "twilio_protocol_rejected",
                error_code=error_code,
            )
            await websocket.close(code=1003)
        except Exception as error:
            session.fail("CALL_BRIDGE_FAILED", CallTerminationReason.INTERNAL_ERROR)
            _app_logger(app).error(
                "call_bridge_failed",
                error_type=type(error).__name__,
            )
            await websocket.close(code=1011)
        finally:
            if tasks:
                await _cancel_tasks(
                    tasks,
                    timeout_seconds=persistence_config.cleanup_timeout_seconds,
                    logger=_app_logger(app),
                )
            await _close_call_session(session)
            if lease is not None:
                await admission.release(lease)

    return app


async def _cancel_tasks(
    tasks: tuple[asyncio.Task[None], ...],
    *,
    timeout_seconds: float,
    logger: FilteringBoundLogger,
) -> None:
    for task in tasks:
        if not task.done():
            task.cancel()
    done, pending = await asyncio.wait(tasks, timeout=timeout_seconds)
    for task in done:
        if not task.cancelled():
            task.exception()
    if pending:
        logger.error(
            "call_bridge_worker_cancel_timed_out",
            worker_count=len(pending),
        )


async def _close_call_session(session: CallSession) -> None:
    cleanup = asyncio.create_task(session.close(), name="call-session-cleanup")
    try:
        await asyncio.shield(cleanup)
    except asyncio.CancelledError:
        await cleanup
        raise


async def _receive_twilio_media(*, websocket: WebSocket, session: CallSession) -> None:
    while not session.closed:
        text = await websocket.receive_text()
        event = parse_twilio_event(json.loads(text))
        session.handle_event(event)


async def _send_twilio_media(*, websocket: WebSocket, session: CallSession) -> None:
    while True:
        message = await session.next_outbound_message()
        if message is None:
            return
        await websocket.send_json(message.to_json())
        session.record_outbound_sent(message)


async def _receive_demo_startup(
    *,
    websocket: WebSocket,
    config: TwilioRuntimeConfig,
    demo_config: DemoRuntimeConfig,
    admission: DemoAdmissionController,
) -> tuple[ConnectedEvent, StartEvent, DemoCallLease]:
    """Validate startup and claim admission before any persistence or Nova access."""
    async with asyncio.timeout(demo_config.reservation_ttl_seconds):
        connected = parse_twilio_event(json.loads(await websocket.receive_text()))
        started = parse_twilio_event(json.loads(await websocket.receive_text()))
    if not isinstance(connected, ConnectedEvent) or not isinstance(started, StartEvent):
        raise TwilioProtocolError(
            "TWILIO_DEMO_START_SEQUENCE_INVALID",
            "Demo media must begin with connected and start events",
        )
    if started.account_sid != config.twilio_account_sid:
        raise TwilioProtocolError(
            "TWILIO_ACCOUNT_SID_MISMATCH",
            "Twilio start account does not match configured account",
        )
    reservation_token = dict(started.custom_parameters).get("demoReservation")
    if reservation_token is None:
        raise TwilioProtocolError(
            "TWILIO_DEMO_RESERVATION_MISSING",
            "Demo reservation is missing",
        )
    lease = await admission.claim(reservation_token)
    if lease is None:
        raise TwilioProtocolError(
            "TWILIO_DEMO_RESERVATION_INVALID",
            "Demo reservation is invalid or expired",
        )
    return connected, started, lease


async def _enforce_demo_duration(
    *,
    session: CallSession,
    duration_seconds: float,
) -> None:
    await asyncio.sleep(duration_seconds)
    session.disconnect(CallTerminationReason.DEMO_TIME_LIMIT)
    await session.close()


async def _load_demo_persona_options(app: FastAPI) -> tuple[tuple[str, str], ...]:
    store = _app_persistence_store(app)
    options: list[tuple[str, str]] = []
    for choice in _app_demo_config(app).persona_choices:
        persona = await asyncio.to_thread(store.get_persona, choice.persona_id)
        if persona is None:
            raise PersistenceError("DEMO_PERSONA_UNAVAILABLE")
        options.append((choice.digit, persona.name))
    return tuple(options)


def _demo_rejection_response(*, rate_limited: bool = False) -> Response:
    return Response(
        content=build_demo_rejection_twiml(rate_limited=rate_limited),
        media_type="application/xml",
    )


def _app_config(app: FastAPI) -> TwilioRuntimeConfig:
    return cast(TwilioRuntimeConfig, app.state.config)


def _app_demo_config(app: FastAPI) -> DemoRuntimeConfig:
    return cast(DemoRuntimeConfig, app.state.demo_config)


def _app_demo_admission(app: FastAPI) -> DemoAdmissionController:
    return cast(DemoAdmissionController, app.state.demo_admission)


def _demo_identifier_secret(config: TwilioRuntimeConfig) -> str:
    return config.twilio_auth_token or config.twilio_account_sid


def _app_logger(app: FastAPI) -> FilteringBoundLogger:
    return cast(FilteringBoundLogger, app.state.logger)


def _app_nova_config(app: FastAPI) -> NovaRuntimeConfig:
    return cast(NovaRuntimeConfig, app.state.nova_config)


def _nova_transport_factory(app: FastAPI) -> NovaTransportFactory:
    return cast(NovaTransportFactory, app.state.nova_transport_factory)


def _app_persistence_config(app: FastAPI) -> PersistenceRuntimeConfig:
    return cast(PersistenceRuntimeConfig, app.state.persistence_config)


def _app_readiness_config(app: FastAPI) -> ReadinessRuntimeConfig:
    return cast(ReadinessRuntimeConfig, app.state.readiness_config)


async def _app_readiness_checker(app: FastAPI) -> ReadinessChecker:
    checker = cast(ReadinessChecker | None, app.state.readiness_checker)
    if checker is not None:
        return checker
    factory = cast(ReadinessFactory, app.state.readiness_factory)
    checker = await asyncio.to_thread(
        factory,
        _app_readiness_config(app),
        _app_persistence_config(app),
    )
    app.state.readiness_checker = checker
    return checker


def _app_persistence_store(app: FastAPI) -> PersistenceStore:
    return cast(PersistenceStore, app.state.persistence_store)


def _app_telemetry(app: FastAPI) -> TelemetryService:
    return cast(TelemetryService, app.state.telemetry)


def _single_value_form(form: FormData) -> Mapping[str, str]:
    values: dict[str, str] = {}
    for key, value in form.multi_items():
        if isinstance(value, str):
            values[key] = value
    return values


def _required_auth_token(config: TwilioRuntimeConfig) -> str:
    if config.twilio_auth_token is None:
        raise ConfigurationError("TWILIO_AUTH_TOKEN is required")
    return config.twilio_auth_token


app = create_app()
