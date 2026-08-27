"""Unit tests for deterministic Nova event protocol behavior."""

from __future__ import annotations

import base64
import json
from typing import Any

import pytest

from realtime_voice_agent.config import NovaRuntimeConfig
from realtime_voice_agent.nova.events import (
    CompletionEnded,
    InterruptionStarted,
    NovaEventIds,
    NovaEventParser,
    NovaProtocolError,
    NovaSessionState,
    OutputAudio,
    build_audio_content_start,
    build_audio_input,
    build_initialization_events,
    build_interactive_text_events,
    transition_state,
    validate_pcm16_chunk,
)
from realtime_voice_agent.transcript import FinalTranscript, TranscriptSpeaker


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


@pytest.fixture
def ids() -> NovaEventIds:
    return NovaEventIds(
        prompt_name="prompt-id",
        system_content_name="system-id",
        audio_content_name="audio-id",
    )


def decode_event(payload: bytes) -> dict[str, Any]:
    return json.loads(payload)


def test_initialization_event_order_and_output_contract(
    config: NovaRuntimeConfig,
    ids: NovaEventIds,
) -> None:
    events = build_initialization_events(
        ids=ids,
        config=config,
        system_prompt="A short safe prompt.",
    )

    event_names = [next(iter(decode_event(payload)["event"])) for payload in events]
    prompt_start = decode_event(events[1])["event"]["promptStart"]

    assert event_names == [
        "sessionStart",
        "promptStart",
        "contentStart",
        "textInput",
        "contentEnd",
    ]
    assert prompt_start["audioOutputConfiguration"] == {
        "mediaType": "audio/lpcm",
        "sampleRateHertz": 24_000,
        "sampleSizeBits": 16,
        "channelCount": 1,
        "voiceId": "matthew",
        "encoding": "base64",
        "audioType": "SPEECH",
    }


def test_interactive_text_turn_prompts_spoken_user_response_without_audio() -> None:
    events = build_interactive_text_events(
        prompt_name="prompt-id",
        text="Hello",
    )

    decoded = [decode_event(payload)["event"] for payload in events]
    event_names = [next(iter(event)) for event in decoded]
    content_start = decoded[0]["contentStart"]
    text_input = decoded[1]["textInput"]
    content_end = decoded[2]["contentEnd"]

    assert event_names == ["contentStart", "textInput", "contentEnd"]
    assert content_start["promptName"] == "prompt-id"
    assert content_start["type"] == "TEXT"
    assert content_start["interactive"] is True
    assert content_start["role"] == "USER"
    assert text_input["content"] == "Hello"
    assert text_input["contentName"] == content_start["contentName"]
    assert content_end["contentName"] == content_start["contentName"]


def test_audio_content_start_has_explicit_input_contract(
    config: NovaRuntimeConfig,
    ids: NovaEventIds,
) -> None:
    event = decode_event(build_audio_content_start(ids=ids, config=config))

    assert event["event"]["contentStart"]["audioInputConfiguration"] == {
        "mediaType": "audio/lpcm",
        "sampleRateHertz": 16_000,
        "sampleSizeBits": 16,
        "channelCount": 1,
        "audioType": "SPEECH",
        "encoding": "base64",
    }


def test_audio_input_round_trips_raw_pcm_without_headers(ids: NovaEventIds) -> None:
    pcm = b"\x00\x00\xff\x7f\x00\x80"

    event = decode_event(build_audio_input(ids=ids, pcm16le_16khz=pcm))
    encoded = event["event"]["audioInput"]["content"]

    assert base64.b64decode(encoded, validate=True) == pcm


@pytest.mark.parametrize("audio", [b"", b"\x00", b"\x00\x00\x01"])
def test_pcm16_validation_rejects_empty_or_partial_samples(audio: bytes) -> None:
    with pytest.raises(NovaProtocolError, match="PCM16"):
        validate_pcm16_chunk(audio)


def test_parse_server_audio_validates_and_decodes_pcm() -> None:
    pcm = b"\x00\x00\x01\x00"
    payload = json.dumps(
        {"event": {"audioOutput": {"content": base64.b64encode(pcm).decode("ascii")}}}
    ).encode()

    event = NovaEventParser().parse(payload)

    assert event == OutputAudio(pcm16le_24khz=pcm)


def test_parse_server_audio_rejects_invalid_base64() -> None:
    payload = b'{"event":{"audioOutput":{"content":"not base64!"}}}'

    with pytest.raises(NovaProtocolError, match="valid base64"):
        NovaEventParser().parse(payload)


def test_parse_server_completion() -> None:
    assert NovaEventParser().parse(b'{"event":{"completionEnd":{}}}') == CompletionEnded()


def test_speculative_transcript_is_ignored() -> None:
    parser = NovaEventParser()

    assert parser.parse(_content_start("speculative", "ASSISTANT", "SPECULATIVE")) is None
    assert parser.parse(_text_output("speculative", "sensitive draft")) is None
    assert parser.parse(_content_end("speculative")) is None


@pytest.mark.parametrize(
    ("role", "speaker"),
    [
        ("USER", TranscriptSpeaker.CALLER),
        ("ASSISTANT", TranscriptSpeaker.AGENT),
    ],
)
def test_final_transcript_is_emitted_once_on_content_end(
    role: str,
    speaker: TranscriptSpeaker,
) -> None:
    parser = NovaEventParser()

    assert parser.parse(_content_start("final-1", role, "FINAL")) is None
    assert parser.parse(_text_output("final-1", "first ")) is None
    assert parser.parse(_text_output("final-1", "sentence")) is None

    assert parser.parse(_content_end("final-1")) == FinalTranscript(
        speaker=speaker,
        text="first sentence",
        source_event_id="final-1",
    )
    assert parser.parse(_content_end("final-1")) is None


def test_final_interruption_control_marker_is_not_a_transcript() -> None:
    parser = NovaEventParser()

    assert parser.parse(_content_start("control-1", "ASSISTANT", "FINAL")) is None
    assert parser.parse(_text_output("control-1", '{ "interrupted" : true }')) is None

    assert parser.parse(_content_end("control-1")) is None


def test_official_interruption_event_is_emitted_and_discards_transcript_buffer() -> None:
    parser = NovaEventParser()

    assert parser.parse(_content_start("interrupted-1", "ASSISTANT", "FINAL")) is None
    assert parser.parse(_text_output("interrupted-1", "unfinished response")) is None

    assert parser.parse(
        _content_end("interrupted-1", stop_reason="INTERRUPTED")
    ) == InterruptionStarted(source_event_id="interrupted-1")
    assert parser.parse(_content_end("interrupted-1")) is None


def _content_start(
    content_id: str,
    role: str,
    generation_stage: str,
) -> bytes:
    return json.dumps(
        {
            "event": {
                "contentStart": {
                    "contentId": content_id,
                    "type": "TEXT",
                    "role": role,
                    "additionalModelFields": json.dumps({"generationStage": generation_stage}),
                }
            }
        }
    ).encode()


def _text_output(content_id: str, content: str) -> bytes:
    return json.dumps(
        {
            "event": {
                "textOutput": {
                    "contentId": content_id,
                    "content": content,
                }
            }
        }
    ).encode()


def _content_end(content_id: str, *, stop_reason: str | None = None) -> bytes:
    content_end = {
        "contentId": content_id,
        "type": "TEXT",
    }
    if stop_reason is not None:
        content_end["stopReason"] = stop_reason
    return json.dumps({"event": {"contentEnd": content_end}}).encode()


def test_state_machine_accepts_valid_lifecycle() -> None:
    state = NovaSessionState.NEW
    for target in (
        NovaSessionState.STARTING,
        NovaSessionState.ACTIVE,
        NovaSessionState.CLOSING,
        NovaSessionState.CLOSED,
    ):
        state = transition_state(state, target)

    assert state is NovaSessionState.CLOSED


def test_state_machine_rejects_invalid_transition() -> None:
    with pytest.raises(NovaProtocolError, match="invalid Nova session transition"):
        transition_state(NovaSessionState.NEW, NovaSessionState.ACTIVE)
