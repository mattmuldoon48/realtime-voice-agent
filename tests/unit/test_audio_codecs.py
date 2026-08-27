"""Unit tests for raw Twilio mu-law codec helpers."""

from __future__ import annotations

import base64
import struct

import pytest

from realtime_voice_agent.audio.codecs import (
    AudioCodecError,
    Pcm16MonoResampler,
    decode_twilio_mulaw_payload,
    encode_twilio_mulaw_payload,
)


def test_twilio_mulaw_payload_round_trips_pcm_without_container_header() -> None:
    pcm16 = (1_000).to_bytes(2, "little", signed=True) * 160

    payload = encode_twilio_mulaw_payload(pcm16)
    raw = base64.b64decode(payload, validate=True)
    decoded = decode_twilio_mulaw_payload(payload)

    assert raw[:4] != b"RIFF"
    assert len(raw) == len(pcm16) // 2
    assert len(decoded) == len(pcm16)
    assert any(sample != 0 for sample in decoded)


@pytest.mark.parametrize("payload", ["not base64", ""])
def test_invalid_twilio_payload_is_rejected(payload: str) -> None:
    with pytest.raises(AudioCodecError):
        decode_twilio_mulaw_payload(payload)


def test_odd_length_pcm16_cannot_be_sent_to_twilio() -> None:
    with pytest.raises(AudioCodecError, match="ODD_LENGTH_PCM16_PAYLOAD"):
        encode_twilio_mulaw_payload(b"\x00")


def test_known_mulaw_fixture_decodes_to_expected_pcm_samples() -> None:
    decoded = decode_twilio_mulaw_payload("/38AgA==")

    assert struct.unpack("<hhhh", decoded) == (0, 0, -32124, 32124)


def test_chunked_resampling_matches_continuous_resampling() -> None:
    pcm16_8khz = b"".join(
        ((index % 200) - 100).to_bytes(2, "little", signed=True) for index in range(320)
    )
    continuous = Pcm16MonoResampler(
        source_rate_hz=8_000,
        target_rate_hz=16_000,
    ).convert(pcm16_8khz)
    chunked_converter = Pcm16MonoResampler(
        source_rate_hz=8_000,
        target_rate_hz=16_000,
    )

    chunked = b"".join(
        (
            chunked_converter.convert(pcm16_8khz[:320]),
            chunked_converter.convert(pcm16_8khz[320:]),
        )
    )

    assert chunked == continuous
    assert len(chunked) > len(pcm16_8khz)


def test_chunked_nova_output_resampling_matches_continuous_resampling() -> None:
    pcm16_24khz = b"".join(
        ((index % 400) - 200).to_bytes(2, "little", signed=True) for index in range(960)
    )
    continuous = Pcm16MonoResampler(
        source_rate_hz=24_000,
        target_rate_hz=8_000,
    ).convert(pcm16_24khz)
    chunked_converter = Pcm16MonoResampler(
        source_rate_hz=24_000,
        target_rate_hz=8_000,
    )

    chunked = b"".join(
        (
            chunked_converter.convert(pcm16_24khz[:960]),
            chunked_converter.convert(pcm16_24khz[960:]),
        )
    )

    assert chunked == continuous
    assert len(chunked) < len(pcm16_24khz)
