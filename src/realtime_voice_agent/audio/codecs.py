"""Raw Twilio G.711 mu-law conversion helpers."""

from __future__ import annotations

import audioop
import base64
import binascii
from typing import Final

TWILIO_SAMPLE_RATE: Final = 8_000
TWILIO_CHANNELS: Final = 1
TWILIO_ENCODING: Final = "audio/x-mulaw"
PCM16_SAMPLE_WIDTH_BYTES: Final = 2
type _RatecvState = tuple[int, tuple[tuple[int, int], ...]]


class Pcm16MonoResampler:
    """Stateful raw PCM16 mono sample-rate converter for one audio stream."""

    def __init__(self, *, source_rate_hz: int, target_rate_hz: int) -> None:
        if source_rate_hz <= 0 or target_rate_hz <= 0:
            raise ValueError("sample rates must be positive")
        self._source_rate_hz = source_rate_hz
        self._target_rate_hz = target_rate_hz
        self._state: _RatecvState | None = None

    def convert(self, pcm16: bytes) -> bytes:
        """Convert one aligned chunk while preserving state between calls."""
        if len(pcm16) % PCM16_SAMPLE_WIDTH_BYTES != 0:
            raise AudioCodecError("ODD_LENGTH_PCM16_PAYLOAD")
        if not pcm16:
            raise AudioCodecError("EMPTY_PCM16_PAYLOAD")
        converted, self._state = audioop.ratecv(
            pcm16,
            PCM16_SAMPLE_WIDTH_BYTES,
            TWILIO_CHANNELS,
            self._source_rate_hz,
            self._target_rate_hz,
            self._state,
        )
        return converted


class AudioCodecError(ValueError):
    """Raised when a media payload cannot be safely converted."""


def decode_twilio_mulaw_payload(payload: str) -> bytes:
    """Decode one Twilio base64 raw mu-law payload into PCM16 mono 8 kHz bytes."""
    try:
        raw_mulaw = base64.b64decode(payload.encode("ascii"), validate=True)
    except (UnicodeEncodeError, binascii.Error) as exc:
        raise AudioCodecError("INVALID_BASE64_MEDIA_PAYLOAD") from exc
    if not raw_mulaw:
        raise AudioCodecError("EMPTY_MEDIA_PAYLOAD")
    return audioop.ulaw2lin(raw_mulaw, PCM16_SAMPLE_WIDTH_BYTES)


def encode_twilio_mulaw_payload(pcm16_8khz: bytes) -> str:
    """Encode PCM16 mono 8 kHz bytes into one Twilio base64 raw mu-law payload."""
    if len(pcm16_8khz) % PCM16_SAMPLE_WIDTH_BYTES != 0:
        raise AudioCodecError("ODD_LENGTH_PCM16_PAYLOAD")
    if not pcm16_8khz:
        raise AudioCodecError("EMPTY_PCM16_PAYLOAD")
    raw_mulaw = audioop.lin2ulaw(pcm16_8khz, PCM16_SAMPLE_WIDTH_BYTES)
    return base64.b64encode(raw_mulaw).decode("ascii")
