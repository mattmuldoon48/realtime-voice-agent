"""Twilio Voice webhook helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import ceil
from typing import cast

from twilio.request_validator import RequestValidator  # type: ignore[import-untyped]
from twilio.twiml.voice_response import (  # type: ignore[import-untyped]
    Connect,
    Gather,
    VoiceResponse,
)


class TwilioSignatureError(PermissionError):
    """Raised when a Twilio webhook signature is invalid."""


def build_connect_stream_twiml(
    *,
    media_ws_url: str,
    custom_parameters: Mapping[str, str] | None = None,
    spoken_intro: str | None = None,
) -> str:
    """Build TwiML that introduces and opens a bidirectional Media Stream."""
    response = VoiceResponse()
    if spoken_intro is not None:
        response.say(spoken_intro)
    connect = Connect()
    stream = connect.stream(url=media_ws_url)
    for name, value in (custom_parameters or {}).items():
        stream.parameter(name=name, value=value)
    response.append(connect)
    return str(response)


def build_demo_menu_twiml(
    *,
    selection_url: str,
    persona_options: Sequence[tuple[str, str]],
    max_duration_seconds: float,
    invalid_selection: bool = False,
) -> str:
    """Build the bounded inbound-only DTMF menu for configured personas."""
    response = VoiceResponse()
    if not invalid_selection:
        response.say(
            "This is a portfolio demonstration. Do not provide sensitive personal, medical, "
            "financial, authentication, or account information."
        )
        response.say(f"This call is limited to {ceil(max_duration_seconds / 60)} minutes.")
    gather = Gather(
        input="dtmf",
        num_digits=1,
        timeout=6,
        action=selection_url,
        method="POST",
    )
    if invalid_selection:
        gather.say("That selection was not recognized.")
    for digit, name in persona_options:
        gather.say(f"Press {digit} for {name}.")
    response.append(gather)
    response.say("No selection was received. Please call again when you are ready.")
    response.hangup()
    return str(response)


def build_demo_rejection_twiml(*, rate_limited: bool = False) -> str:
    """Build a short response that does not disclose internal budget or capacity state."""
    response = VoiceResponse()
    if rate_limited:
        response.say("This demo's per-caller usage limit has been reached. Please try again later.")
    else:
        response.say("The portfolio demo is currently unavailable. Please try again later.")
    response.hangup()
    return str(response)


def validate_twilio_signature(
    *,
    auth_token: str,
    public_url: str,
    form: Mapping[str, str],
    signature: str | None,
) -> None:
    """Validate the webhook against the externally visible URL Twilio called."""
    if not signature:
        raise TwilioSignatureError("Missing X-Twilio-Signature")
    validator = RequestValidator(auth_token)
    is_valid = cast(bool, validator.validate(public_url, dict(form), signature))
    if not is_valid:
        raise TwilioSignatureError("Invalid X-Twilio-Signature")


def validate_twilio_websocket_signature(
    *,
    auth_token: str,
    public_url: str,
    signature: str | None,
) -> None:
    """Validate a WebSocket handshake against its exact configured public URL."""
    validate_twilio_signature(
        auth_token=auth_token,
        public_url=public_url,
        form={},
        signature=signature,
    )
