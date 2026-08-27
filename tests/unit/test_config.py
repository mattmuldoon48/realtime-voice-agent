"""Unit tests for application configuration resolution."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from pydantic import ValidationError

from realtime_voice_agent.config import AppSettings, ConfigurationError


@dataclass
class FakeSession:
    region_name: str | None


def test_explicit_region_and_profile_are_preserved() -> None:
    settings = AppSettings(
        _env_file=None,
        aws_region="us-west-2",
        aws_profile="nova-demo",
    )

    config = settings.to_runtime_config(
        session_factory=lambda _profile: pytest.fail("profile lookup was not expected")
    )

    assert config.region == "us-west-2"
    assert config.profile == "nova-demo"
    assert config.model_id == "amazon.nova-2-sonic-v1:0"
    assert config.input_sample_rate == 16_000
    assert config.output_sample_rate == 24_000
    assert config.channels == 1
    assert config.sample_width_bytes == 2
    assert config.session_rotation_seconds == 240.0


def test_region_falls_back_to_selected_profile_configuration() -> None:
    selected_profiles: list[str | None] = []
    settings = AppSettings(
        _env_file=None,
        aws_region=None,
        aws_profile="nova-demo",
    )

    config = settings.to_runtime_config(
        session_factory=lambda profile: (
            selected_profiles.append(profile) or FakeSession("eu-north-1")
        )
    )

    assert config.region == "eu-north-1"
    assert selected_profiles == ["nova-demo"]


def test_blank_profile_is_normalized_to_standard_default_chain() -> None:
    settings = AppSettings(
        _env_file=None,
        aws_region="ap-northeast-1",
        aws_profile="   ",
    )

    assert settings.aws_profile is None


def test_missing_region_is_rejected() -> None:
    settings = AppSettings(_env_file=None, aws_region=None)

    with pytest.raises(ConfigurationError, match="AWS_REGION is required"):
        settings.to_runtime_config(session_factory=lambda _profile: FakeSession(None))


def test_invalid_region_is_rejected_without_echoing_secrets() -> None:
    settings = AppSettings(_env_file=None, aws_region="not-a-region")

    with pytest.raises(ConfigurationError, match="not a valid AWS Region"):
        settings.to_runtime_config()


def test_nova_rotation_threshold_keeps_thirty_second_expiry_margin() -> None:
    with pytest.raises(ValidationError, match="must be at most 450"):
        AppSettings(
            _env_file=None,
            aws_region="us-west-2",
            nova_session_rotation_seconds=451,
        )


def test_persistence_retry_attempts_are_bounded() -> None:
    with pytest.raises(ValidationError, match="must be at most 5"):
        AppSettings(
            _env_file=None,
            aws_region="us-west-2",
            persistence_max_attempts=6,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("nova_model_id", "amazon.nova-sonic-v1:0"),
        ("nova_input_sample_rate", 8_000),
        ("nova_output_sample_rate", 16_000),
        ("nova_channels", 2),
        ("nova_sample_width_bytes", 1),
    ],
)
def test_fixed_nova_contract_rejects_drift(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        AppSettings(
            _env_file=None,
            aws_region="us-west-2",
            **{field: value},
        )


def test_fixed_nova_contract_accepts_environment_integer_strings() -> None:
    settings = AppSettings(
        _env_file=None,
        aws_region="us-west-2",
        nova_input_sample_rate="16000",
        nova_output_sample_rate="24000",
        nova_channels="1",
        nova_sample_width_bytes="2",
    )

    config = settings.to_runtime_config()

    assert config.input_sample_rate == 16_000
    assert config.output_sample_rate == 24_000
    assert config.channels == 1
    assert config.sample_width_bytes == 2


def test_persistence_runtime_uses_shared_aws_resolution_and_bounded_defaults() -> None:
    settings = AppSettings(
        _env_file=None,
        aws_region="us-west-2",
        aws_profile="nova-demo",
        personas_table=" personas ",
        sessions_table="sessions",
        transcripts_table="turns",
    )

    config = settings.to_persistence_runtime_config(
        session_factory=lambda _profile: pytest.fail("profile lookup was not expected")
    )

    assert config.region == "us-west-2"
    assert config.profile == "nova-demo"
    assert config.personas_table == "personas"
    assert config.sessions_table == "sessions"
    assert config.transcripts_table == "turns"
    assert config.transcript_retention_days == 7
    assert config.queue_max_events == 100
    assert config.cleanup_timeout_seconds == 5.0
    assert config.max_attempts == 3
    assert config.retry_base_delay_seconds == 0.1


def test_readiness_runtime_uses_shared_aws_resolution_and_bounded_timeout() -> None:
    settings = AppSettings(
        _env_file=None,
        aws_region="us-west-2",
        aws_profile="nova-demo",
        personas_table="personas",
        sessions_table="sessions",
        transcripts_table="turns",
        readiness_timeout_seconds="3",
    )

    config = settings.to_readiness_runtime_config(
        session_factory=lambda _profile: pytest.fail("profile lookup was not expected")
    )

    assert config.region == "us-west-2"
    assert config.profile == "nova-demo"
    assert config.personas_table == "personas"
    assert config.sessions_table == "sessions"
    assert config.transcripts_table == "turns"
    assert config.timeout_seconds == 3.0


def test_observability_runtime_uses_bounded_cloudwatch_defaults() -> None:
    settings = AppSettings(
        _env_file=None,
        aws_profile="nova-demo",
        app_env="smoke",
        cloudwatch_enabled="true",
        telemetry_queue_max_events="250",
        telemetry_batch_max_items="50",
    )

    config = settings.to_observability_runtime_config(
        session_factory=lambda _profile: FakeSession("us-east-1")
    )

    assert config.enabled is True
    assert config.region == "us-east-1"
    assert config.profile == "nova-demo"
    assert config.environment == "smoke"
    assert config.log_group == "/realtime-voice-agent/application"
    assert config.log_stream == "local-smoke"
    assert config.metric_namespace == "RealtimeVoiceAgent/VoiceAgent"
    assert config.queue_max_events == 250
    assert config.batch_max_items == 50
    assert config.max_attempts == 2


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("app_env", "per-call/session-id", "APP_ENV"),
        ("cloudwatch_log_group", "relative-group", "CLOUDWATCH_LOG_GROUP"),
        ("telemetry_batch_max_items", 1_001, "at most 1000"),
    ],
)
def test_observability_settings_reject_unbounded_or_invalid_values(
    field: str,
    value: object,
    match: str,
) -> None:
    with pytest.raises(ValidationError, match=match):
        AppSettings(_env_file=None, aws_region="us-east-1", **{field: value})


def test_phone_number_storage_cannot_be_enabled() -> None:
    with pytest.raises(ValidationError, match="STORE_PHONE_NUMBERS must remain false"):
        AppSettings(
            _env_file=None,
            aws_region="us-east-1",
            store_phone_numbers=True,
        )


def test_twilio_runtime_derives_secure_media_url_from_public_base() -> None:
    settings = AppSettings(
        _env_file=None,
        public_base_url="https://example.ngrok.app/",
        twilio_auth_token="not-a-real-token",
        twilio_account_sid="test-account-sid",
    )

    config = settings.to_twilio_runtime_config()

    assert config.public_base_url == "https://example.ngrok.app"
    assert config.public_media_ws_url == "wss://example.ngrok.app/media"
    assert config.twilio_account_sid == "test-account-sid"
    assert config.validate_signatures is True
    assert config.twilio_audio_queue_max_frames == 100
    assert config.twilio_outbound_queue_max_frames == 200


def test_twilio_runtime_allows_explicit_secure_media_url() -> None:
    settings = AppSettings(
        _env_file=None,
        public_base_url="https://voice.example.test",
        public_media_ws_url="wss://media.example.test/custom",
        twilio_account_sid="test-account-sid",
        twilio_validate_signatures=False,
    )

    config = settings.to_twilio_runtime_config()

    assert config.public_media_ws_url == "wss://media.example.test/custom"
    assert config.twilio_auth_token is None


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("public_base_url", None, "PUBLIC_BASE_URL"),
        ("public_base_url", "http://example.test", "PUBLIC_BASE_URL must use https"),
        ("public_media_ws_url", "ws://example.test/media", "PUBLIC_MEDIA_WS_URL must use wss"),
    ],
)
def test_twilio_runtime_rejects_invalid_public_urls(
    field: str,
    value: str | None,
    match: str,
) -> None:
    values = {
        "_env_file": None,
        "public_base_url": "https://example.test",
        "twilio_account_sid": "test-account-sid",
        "twilio_auth_token": "not-a-real-token",
        field: value,
    }
    settings = AppSettings(**values)

    with pytest.raises(ConfigurationError, match=match):
        settings.to_twilio_runtime_config()


def test_twilio_signature_token_required_only_when_validation_is_enabled() -> None:
    settings = AppSettings(
        _env_file=None,
        public_base_url="https://example.test",
        twilio_account_sid="test-account-sid",
        twilio_validate_signatures=True,
    )

    with pytest.raises(ConfigurationError, match="TWILIO_AUTH_TOKEN"):
        settings.to_twilio_runtime_config()


def test_twilio_account_sid_is_required_for_start_event_defense() -> None:
    settings = AppSettings(
        _env_file=None,
        public_base_url="https://example.test",
        twilio_validate_signatures=False,
    )

    with pytest.raises(ConfigurationError, match="TWILIO_ACCOUNT_SID"):
        settings.to_twilio_runtime_config()


def test_demo_runtime_defaults_to_four_configured_personas_and_five_minutes() -> None:
    config = AppSettings(_env_file=None).to_demo_runtime_config()

    assert config.enabled is False
    assert [(choice.digit, choice.persona_id) for choice in config.persona_choices] == [
        ("1", "care-coordinator"),
        ("2", "financial-services-assistant"),
        ("3", "travel-concierge"),
        ("4", "history-guide"),
    ]
    assert config.max_call_duration_seconds == 300
    assert config.rate_limit_max_calls == 3
    assert config.global_concurrency_limit == 2
    assert config.budget_max_calls == 50
    assert config.persist_transcripts is False


def test_demo_runtime_limits_and_persona_mapping_are_environment_configurable() -> None:
    settings = AppSettings(
        _env_file=None,
        demo_mode_enabled="true",
        demo_persona_choices="7:history-guide,9:travel-concierge",
        demo_max_call_duration_seconds="120",
        demo_rate_limit_max_calls="4",
        demo_rate_limit_window_seconds="900",
        demo_global_concurrency_limit="3",
        demo_budget_max_calls="20",
        demo_budget_window_seconds="7200",
        demo_reservation_ttl_seconds="20",
        demo_persist_transcripts="true",
    )

    config = settings.to_demo_runtime_config()

    assert config.enabled is True
    assert [(choice.digit, choice.persona_id) for choice in config.persona_choices] == [
        ("7", "history-guide"),
        ("9", "travel-concierge"),
    ]
    assert config.max_call_duration_seconds == 120
    assert config.rate_limit_max_calls == 4
    assert config.rate_limit_window_seconds == 900
    assert config.global_concurrency_limit == 3
    assert config.budget_max_calls == 20
    assert config.budget_window_seconds == 7200
    assert config.reservation_ttl_seconds == 20
    assert config.persist_transcripts is True


@pytest.mark.parametrize(
    "choices",
    [
        "care-coordinator",
        "0:care-coordinator",
        "1:care-coordinator,1:history-guide",
        "1:care-coordinator,2:care-coordinator",
    ],
)
def test_demo_runtime_rejects_malformed_or_duplicate_persona_choices(choices: str) -> None:
    settings = AppSettings(_env_file=None, demo_persona_choices=choices)

    with pytest.raises(ConfigurationError, match="DEMO_PERSONA_CHOICES"):
        settings.to_demo_runtime_config()


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("demo_max_call_duration_seconds", 1_801, "at most 1800"),
        ("demo_reservation_ttl_seconds", 121, "at most 120"),
        ("demo_global_concurrency_limit", 10_001, "at most 10000"),
    ],
)
def test_demo_runtime_rejects_unbounded_limits(
    field: str,
    value: object,
    match: str,
) -> None:
    with pytest.raises(ValidationError, match=match):
        AppSettings(_env_file=None, **{field: value})
