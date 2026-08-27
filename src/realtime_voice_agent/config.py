"""Typed configuration for local project commands."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final, Literal, Protocol, cast

import boto3  # type: ignore[import-untyped]
from pydantic import PositiveFloat, PositiveInt, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

NOVA_MODEL_ID: Final[Literal["amazon.nova-2-sonic-v1:0"]] = "amazon.nova-2-sonic-v1:0"
NOVA_INPUT_SAMPLE_RATE: Final[Literal[16000]] = 16_000
NOVA_OUTPUT_SAMPLE_RATE: Final[Literal[24000]] = 24_000
NOVA_CHANNELS: Final[Literal[1]] = 1
NOVA_SAMPLE_WIDTH_BYTES: Final[Literal[2]] = 2

_REGION_PATTERN = re.compile(r"^[a-z]{2}(?:-gov)?-[a-z]+-\d+$")
_PERSONA_ID_PATTERN = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?$")


class ConfigurationError(ValueError):
    """Raised when required local configuration cannot be resolved safely."""


class AwsSessionLike(Protocol):
    """Narrow boto session contract used for Region resolution."""

    @property
    def region_name(self) -> str | None: ...


SessionFactory = Callable[[str | None], AwsSessionLike]


@dataclass(frozen=True, slots=True)
class NovaRuntimeConfig:
    """Validated values required by the standalone Nova adapter."""

    region: str
    profile: str | None
    model_id: Literal["amazon.nova-2-sonic-v1:0"]
    voice_id: str
    input_sample_rate: Literal[16000]
    output_sample_rate: Literal[24000]
    channels: Literal[1]
    sample_width_bytes: Literal[2]
    chunk_frames: int
    session_rotation_seconds: float = 240.0


@dataclass(frozen=True, slots=True)
class TwilioRuntimeConfig:
    """Validated values required by Twilio Media Streams."""

    app_env: str
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    service_name: str
    public_base_url: str
    public_media_ws_url: str
    twilio_auth_token: str | None
    twilio_account_sid: str
    validate_signatures: bool
    malformed_media_frame_limit: int
    twilio_audio_queue_max_frames: int
    twilio_outbound_queue_max_frames: int


@dataclass(frozen=True, slots=True)
class DemoPersonaChoice:
    """One DTMF digit mapped to a configured persona identifier."""

    digit: str
    persona_id: str


@dataclass(frozen=True, slots=True)
class DemoRuntimeConfig:
    """Validated public-demo admission, privacy, and cost limits."""

    enabled: bool
    persona_choices: tuple[DemoPersonaChoice, ...]
    max_call_duration_seconds: float
    rate_limit_max_calls: int
    rate_limit_window_seconds: float
    global_concurrency_limit: int
    budget_max_calls: int
    budget_window_seconds: float
    reservation_ttl_seconds: float
    persist_transcripts: bool


@dataclass(frozen=True, slots=True)
class PersistenceRuntimeConfig:
    """Validated DynamoDB table, retention, and worker settings."""

    region: str
    profile: str | None
    personas_table: str
    sessions_table: str
    transcripts_table: str
    transcript_retention_days: int
    store_phone_numbers: bool
    queue_max_events: int
    cleanup_timeout_seconds: float
    max_attempts: int
    retry_base_delay_seconds: float


@dataclass(frozen=True, slots=True)
class ReadinessRuntimeConfig:
    """Validated AWS dependency readiness settings."""

    region: str
    profile: str | None
    personas_table: str
    sessions_table: str
    transcripts_table: str
    timeout_seconds: float


@dataclass(frozen=True, slots=True)
class ObservabilityRuntimeConfig:
    """Validated CloudWatch worker, log, and metric settings."""

    enabled: bool
    region: str
    profile: str | None
    environment: str
    service_name: str
    log_group: str
    log_stream: str
    metric_namespace: Literal["RealtimeVoiceAgent/VoiceAgent"]
    queue_max_events: int
    batch_max_items: int
    flush_interval_seconds: float
    max_attempts: int
    cleanup_timeout_seconds: float


class AppSettings(BaseSettings):
    """Load application settings from environment variables and an ignored local `.env`."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_env: str = "local"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    service_name: str = "realtime-voice-agent"
    aws_region: str | None = None
    aws_profile: str | None = None
    nova_model_id: Literal["amazon.nova-2-sonic-v1:0"] = NOVA_MODEL_ID
    nova_voice_id: str = "matthew"
    nova_input_sample_rate: Literal[16000] = NOVA_INPUT_SAMPLE_RATE
    nova_output_sample_rate: Literal[24000] = NOVA_OUTPUT_SAMPLE_RATE
    nova_channels: Literal[1] = NOVA_CHANNELS
    nova_sample_width_bytes: Literal[2] = NOVA_SAMPLE_WIDTH_BYTES
    nova_audio_chunk_frames: PositiveInt = 512
    nova_session_rotation_seconds: PositiveFloat = 240.0
    twilio_account_sid: str | None = None
    twilio_auth_token: str | None = None
    twilio_validate_signatures: bool = True
    public_base_url: str | None = None
    public_media_ws_url: str | None = None
    demo_mode_enabled: bool = False
    demo_persona_choices: str = (
        "1:care-coordinator,2:financial-services-assistant,3:travel-concierge,4:history-guide"
    )
    demo_max_call_duration_seconds: PositiveFloat = 300.0
    demo_rate_limit_max_calls: PositiveInt = 3
    demo_rate_limit_window_seconds: PositiveFloat = 3_600.0
    demo_global_concurrency_limit: PositiveInt = 2
    demo_budget_max_calls: PositiveInt = 50
    demo_budget_window_seconds: PositiveFloat = 86_400.0
    demo_reservation_ttl_seconds: PositiveFloat = 30.0
    demo_persist_transcripts: bool = False
    malformed_media_frame_limit: PositiveInt = 3
    twilio_audio_queue_max_frames: PositiveInt = 100
    twilio_outbound_queue_max_frames: PositiveInt = 200
    personas_table: str = "RealtimeVoiceAgentPersonas"
    sessions_table: str = "RealtimeVoiceAgentSessions"
    transcripts_table: str = "RealtimeVoiceAgentTranscriptTurns"
    transcript_retention_days: PositiveInt = 7
    store_phone_numbers: bool = False
    persistence_queue_max_events: PositiveInt = 100
    persistence_cleanup_timeout_seconds: PositiveFloat = 5.0
    persistence_max_attempts: PositiveInt = 3
    persistence_retry_base_delay_seconds: PositiveFloat = 0.1
    readiness_timeout_seconds: PositiveFloat = 5.0
    cloudwatch_enabled: bool = False
    cloudwatch_log_group: str = "/realtime-voice-agent/application"
    cloudwatch_log_stream: str = "local"
    cloudwatch_metric_namespace: Literal["RealtimeVoiceAgent/VoiceAgent"] = (
        "RealtimeVoiceAgent/VoiceAgent"
    )
    telemetry_queue_max_events: PositiveInt = 1_000
    telemetry_batch_max_items: PositiveInt = 100
    telemetry_flush_interval_seconds: PositiveFloat = 1.0
    telemetry_max_attempts: PositiveInt = 2
    telemetry_cleanup_timeout_seconds: PositiveFloat = 5.0

    @field_validator(
        "aws_region",
        "aws_profile",
        "twilio_account_sid",
        "twilio_auth_token",
        "public_base_url",
        "public_media_ws_url",
        mode="before",
    )
    @classmethod
    def blank_optional_values_are_unset(cls, value: object) -> object:
        """Treat blank optional environment values as absent."""
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator(
        "nova_input_sample_rate",
        "nova_output_sample_rate",
        "nova_channels",
        "nova_sample_width_bytes",
        mode="before",
    )
    @classmethod
    def fixed_integer_contract_values_allow_env_strings(cls, value: object) -> object:
        """Allow `.env` string values to satisfy fixed integer Nova contract literals."""
        if isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                return value
        return value

    @field_validator("nova_voice_id")
    @classmethod
    def voice_id_must_not_be_blank(cls, value: str) -> str:
        """Reject a blank voice identifier before opening a paid stream."""
        value = value.strip()
        if not value:
            raise ValueError("NOVA_VOICE_ID must not be blank")
        return value

    @field_validator(
        "personas_table",
        "sessions_table",
        "transcripts_table",
        "cloudwatch_log_group",
        "cloudwatch_log_stream",
    )
    @classmethod
    def required_resource_names_are_not_blank(cls, value: str) -> str:
        """Reject blank AWS resource names before constructing clients."""
        value = value.strip()
        if not value:
            raise ValueError("AWS resource names must not be blank")
        return value

    @field_validator("app_env")
    @classmethod
    def environment_is_bounded(cls, value: str) -> str:
        """Restrict the environment metric dimension to a bounded configured label."""
        value = value.strip()
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,31}", value):
            raise ValueError("APP_ENV must be a 1-32 character bounded label")
        return value

    @field_validator("cloudwatch_log_group")
    @classmethod
    def log_group_uses_absolute_name(cls, value: str) -> str:
        """Require the conventional absolute CloudWatch Logs group name."""
        if not value.startswith("/"):
            raise ValueError("CLOUDWATCH_LOG_GROUP must start with '/'")
        return value

    @field_validator("nova_session_rotation_seconds")
    @classmethod
    def nova_rotation_precedes_connection_limit(cls, value: float) -> float:
        """Keep rotation at least thirty seconds below Nova's eight-minute limit."""
        if value > 450:
            raise ValueError("NOVA_SESSION_ROTATION_SECONDS must be at most 450")
        return value

    @field_validator("demo_max_call_duration_seconds")
    @classmethod
    def demo_duration_is_bounded(cls, value: float) -> float:
        if value > 1_800:
            raise ValueError("DEMO_MAX_CALL_DURATION_SECONDS must be at most 1800")
        return value

    @field_validator("demo_reservation_ttl_seconds")
    @classmethod
    def demo_reservation_is_bounded(cls, value: float) -> float:
        if value > 120:
            raise ValueError("DEMO_RESERVATION_TTL_SECONDS must be at most 120")
        return value

    @field_validator(
        "demo_rate_limit_max_calls",
        "demo_global_concurrency_limit",
        "demo_budget_max_calls",
    )
    @classmethod
    def demo_counts_are_bounded(cls, value: int) -> int:
        if value > 10_000:
            raise ValueError("public demo count limits must be at most 10000")
        return value

    @field_validator("persistence_max_attempts")
    @classmethod
    def persistence_attempts_are_bounded(cls, value: int) -> int:
        """Keep per-command DynamoDB retry work strictly bounded."""
        if value > 5:
            raise ValueError("PERSISTENCE_MAX_ATTEMPTS must be at most 5")
        return value

    @field_validator("telemetry_batch_max_items")
    @classmethod
    def telemetry_batch_respects_cloudwatch_limit(cls, value: int) -> int:
        """Keep each PutMetricData request under the service datum limit."""
        if value > 1_000:
            raise ValueError("TELEMETRY_BATCH_MAX_ITEMS must be at most 1000")
        return value

    @field_validator("store_phone_numbers")
    @classmethod
    def phone_number_storage_remains_disabled(cls, value: bool) -> bool:
        """Reject an unsupported privacy override instead of silently storing numbers."""
        if value:
            raise ValueError("STORE_PHONE_NUMBERS must remain false")
        return value

    def to_runtime_config(
        self,
        *,
        session_factory: SessionFactory | None = None,
    ) -> NovaRuntimeConfig:
        """Resolve the AWS Region without hard-coding a fallback Region."""
        factory = session_factory or _new_boto_session
        configured_region = self.aws_region.strip() if self.aws_region else None
        region = configured_region or factory(self.aws_profile).region_name
        if region is None:
            raise ConfigurationError(
                "AWS_REGION is required because no Region exists in the selected AWS profile"
            )
        if not _REGION_PATTERN.fullmatch(region):
            raise ConfigurationError("AWS_REGION is not a valid AWS Region name")

        return NovaRuntimeConfig(
            region=region,
            profile=self.aws_profile,
            model_id=self.nova_model_id,
            voice_id=self.nova_voice_id,
            input_sample_rate=self.nova_input_sample_rate,
            output_sample_rate=self.nova_output_sample_rate,
            channels=self.nova_channels,
            sample_width_bytes=self.nova_sample_width_bytes,
            chunk_frames=self.nova_audio_chunk_frames,
            session_rotation_seconds=self.nova_session_rotation_seconds,
        )

    def to_twilio_runtime_config(self) -> TwilioRuntimeConfig:
        """Resolve safe Twilio Media Streams settings."""
        if self.public_base_url is None:
            raise ConfigurationError("PUBLIC_BASE_URL is required for Twilio webhook validation")
        public_base_url = self.public_base_url.rstrip("/")
        if not public_base_url.startswith("https://"):
            raise ConfigurationError("PUBLIC_BASE_URL must use https")

        public_media_ws_url = self.public_media_ws_url
        if public_media_ws_url is None:
            public_media_ws_url = f"wss://{public_base_url.removeprefix('https://')}/media"
        if not public_media_ws_url.startswith("wss://"):
            raise ConfigurationError("PUBLIC_MEDIA_WS_URL must use wss")

        account_sid = self.twilio_account_sid
        if account_sid is None:
            raise ConfigurationError("TWILIO_ACCOUNT_SID is required")
        auth_token = self.twilio_auth_token
        if self.twilio_validate_signatures and auth_token is None:
            raise ConfigurationError(
                "TWILIO_AUTH_TOKEN is required when TWILIO_VALIDATE_SIGNATURES is true"
            )

        return TwilioRuntimeConfig(
            app_env=self.app_env,
            log_level=self.log_level,
            service_name=self.service_name,
            public_base_url=public_base_url,
            public_media_ws_url=public_media_ws_url,
            twilio_auth_token=auth_token,
            twilio_account_sid=account_sid,
            validate_signatures=self.twilio_validate_signatures,
            malformed_media_frame_limit=self.malformed_media_frame_limit,
            twilio_audio_queue_max_frames=self.twilio_audio_queue_max_frames,
            twilio_outbound_queue_max_frames=self.twilio_outbound_queue_max_frames,
        )

    def to_demo_runtime_config(self) -> DemoRuntimeConfig:
        """Resolve public-demo policy without embedding persona prompts in application logic."""
        return DemoRuntimeConfig(
            enabled=self.demo_mode_enabled,
            persona_choices=_parse_demo_persona_choices(self.demo_persona_choices),
            max_call_duration_seconds=self.demo_max_call_duration_seconds,
            rate_limit_max_calls=self.demo_rate_limit_max_calls,
            rate_limit_window_seconds=self.demo_rate_limit_window_seconds,
            global_concurrency_limit=self.demo_global_concurrency_limit,
            budget_max_calls=self.demo_budget_max_calls,
            budget_window_seconds=self.demo_budget_window_seconds,
            reservation_ttl_seconds=self.demo_reservation_ttl_seconds,
            persist_transcripts=self.demo_persist_transcripts,
        )

    def to_persistence_runtime_config(
        self,
        session_factory: SessionFactory | None = None,
    ) -> PersistenceRuntimeConfig:
        """Resolve DynamoDB settings through the same AWS profile and Region chain."""
        nova = self.to_runtime_config(session_factory=session_factory)
        return PersistenceRuntimeConfig(
            region=nova.region,
            profile=nova.profile,
            personas_table=self.personas_table,
            sessions_table=self.sessions_table,
            transcripts_table=self.transcripts_table,
            transcript_retention_days=self.transcript_retention_days,
            store_phone_numbers=self.store_phone_numbers,
            queue_max_events=self.persistence_queue_max_events,
            cleanup_timeout_seconds=self.persistence_cleanup_timeout_seconds,
            max_attempts=self.persistence_max_attempts,
            retry_base_delay_seconds=self.persistence_retry_base_delay_seconds,
        )

    def to_readiness_runtime_config(
        self,
        session_factory: SessionFactory | None = None,
    ) -> ReadinessRuntimeConfig:
        """Resolve bounded read-only AWS dependency readiness settings."""
        nova = self.to_runtime_config(session_factory=session_factory)
        return ReadinessRuntimeConfig(
            region=nova.region,
            profile=nova.profile,
            personas_table=self.personas_table,
            sessions_table=self.sessions_table,
            transcripts_table=self.transcripts_table,
            timeout_seconds=self.readiness_timeout_seconds,
        )

    def to_observability_runtime_config(
        self,
        session_factory: SessionFactory | None = None,
    ) -> ObservabilityRuntimeConfig:
        """Resolve CloudWatch settings through the standard AWS profile and Region chain."""
        nova = self.to_runtime_config(session_factory=session_factory)
        return ObservabilityRuntimeConfig(
            enabled=self.cloudwatch_enabled,
            region=nova.region,
            profile=nova.profile,
            environment=self.app_env,
            service_name=self.service_name,
            log_group=self.cloudwatch_log_group,
            log_stream=f"{self.cloudwatch_log_stream}-{self.app_env}",
            metric_namespace=self.cloudwatch_metric_namespace,
            queue_max_events=self.telemetry_queue_max_events,
            batch_max_items=self.telemetry_batch_max_items,
            flush_interval_seconds=self.telemetry_flush_interval_seconds,
            max_attempts=self.telemetry_max_attempts,
            cleanup_timeout_seconds=self.telemetry_cleanup_timeout_seconds,
        )


def _parse_demo_persona_choices(value: str) -> tuple[DemoPersonaChoice, ...]:
    choices: list[DemoPersonaChoice] = []
    for entry in value.split(","):
        digit, separator, persona_id = entry.strip().partition(":")
        if (
            separator != ":"
            or len(digit) != 1
            or digit not in "123456789"
            or _PERSONA_ID_PATTERN.fullmatch(persona_id) is None
        ):
            raise ConfigurationError(
                "DEMO_PERSONA_CHOICES must contain comma-separated DIGIT:persona-id entries"
            )
        choices.append(DemoPersonaChoice(digit=digit, persona_id=persona_id))
    digits = {choice.digit for choice in choices}
    persona_ids = {choice.persona_id for choice in choices}
    if not choices or len(digits) != len(choices) or len(persona_ids) != len(choices):
        raise ConfigurationError("DEMO_PERSONA_CHOICES must contain unique digits and persona IDs")
    return tuple(choices)


def _new_boto_session(profile_name: str | None) -> AwsSessionLike:
    """Create a standard-chain boto session without materializing access keys."""
    return cast(AwsSessionLike, boto3.Session(profile_name=profile_name))
