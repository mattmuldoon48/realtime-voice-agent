"""Unit tests for Twilio Media Streams event parsing."""

from __future__ import annotations

import pytest

from realtime_voice_agent.telephony.events import (
    MediaEvent,
    StartEvent,
    TwilioClearCommand,
    TwilioMarkCommand,
    TwilioMediaCommand,
    TwilioProtocolError,
    parse_twilio_event,
)


def valid_start_event() -> dict[str, object]:
    return {
        "event": "start",
        "sequenceNumber": "1",
        "start": {
            "streamSid": "MZ00000000000000000000000000000000",
            "callSid": "CA00000000000000000000000000000000",
            "accountSid": "test-account-sid",
            "mediaFormat": {"encoding": "audio/x-mulaw", "sampleRate": 8000, "channels": 1},
        },
    }


def test_start_event_validates_required_media_format() -> None:
    event = parse_twilio_event(valid_start_event())

    assert isinstance(event, StartEvent)
    assert event.sequence_number == 1
    assert event.account_sid == "test-account-sid"
    assert event.media_format.encoding == "audio/x-mulaw"
    assert event.media_format.sample_rate == 8_000
    assert event.media_format.channels == 1


def test_start_event_retains_only_bounded_custom_parameter_pairs() -> None:
    message = valid_start_event()
    start = message["start"]
    assert isinstance(start, dict)
    start["customParameters"] = {"demoReservation": "opaque-token"}

    event = parse_twilio_event(message)

    assert isinstance(event, StartEvent)
    assert event.custom_parameters == (("demoReservation", "opaque-token"),)


@pytest.mark.parametrize(
    "custom_parameters",
    [
        {"demoReservation": ""},
        {"demoReservation": "x" * 257},
        {f"key-{index}": "value" for index in range(17)},
    ],
)
def test_start_event_rejects_unbounded_custom_parameters(
    custom_parameters: dict[str, str],
) -> None:
    message = valid_start_event()
    start = message["start"]
    assert isinstance(start, dict)
    start["customParameters"] = custom_parameters

    with pytest.raises(TwilioProtocolError) as error:
        parse_twilio_event(message)

    assert error.value.code == "TWILIO_CUSTOM_PARAMETERS_INVALID"


@pytest.mark.parametrize(
    ("field", "value", "code"),
    [
        ("encoding", "audio/pcm", "TWILIO_MEDIA_ENCODING_UNSUPPORTED"),
        ("sampleRate", 16000, "TWILIO_MEDIA_SAMPLE_RATE_UNSUPPORTED"),
        ("channels", 2, "TWILIO_MEDIA_CHANNELS_UNSUPPORTED"),
    ],
)
def test_start_event_rejects_media_format_drift(field: str, value: object, code: str) -> None:
    message = valid_start_event()
    start = message["start"]
    assert isinstance(start, dict)
    media_format = start["mediaFormat"]
    assert isinstance(media_format, dict)
    media_format[field] = value

    with pytest.raises(TwilioProtocolError) as error:
        parse_twilio_event(message)

    assert error.value.code == code
    assert str(value) not in str(error.value)


def test_media_event_preserves_ordering_fields_without_decoding_audio() -> None:
    event = parse_twilio_event(
        {
            "event": "media",
            "sequenceNumber": "2",
            "media": {
                "track": "inbound",
                "chunk": "7",
                "timestamp": "140",
                "payload": "/w==",
            },
        }
    )

    assert isinstance(event, MediaEvent)
    assert event.sequence_number == 2
    assert event.chunk == 7
    assert event.timestamp_ms == 140
    assert event.payload == "/w=="


def test_outbound_media_mark_and_clear_commands_have_exact_twilio_shapes() -> None:
    media = TwilioMediaCommand(
        stream_sid="MZ00000000000000000000000000000000",
        payload="/w==",
        mulaw_bytes=1,
        generation=2,
    )
    mark = TwilioMarkCommand(
        stream_sid=media.stream_sid,
        name="response-3",
        generation=2,
    )
    clear = TwilioClearCommand(stream_sid=media.stream_sid, generation=3)

    assert media.to_json() == {
        "event": "media",
        "streamSid": media.stream_sid,
        "media": {"payload": "/w=="},
    }
    assert mark.to_json() == {
        "event": "mark",
        "streamSid": media.stream_sid,
        "mark": {"name": "response-3"},
    }
    assert clear.to_json() == {
        "event": "clear",
        "streamSid": media.stream_sid,
    }
