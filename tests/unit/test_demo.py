"""Behavioral tests for public-demo admission and privacy safeguards."""

from __future__ import annotations

from dataclasses import replace

import pytest

from realtime_voice_agent.config import DemoRuntimeConfig
from realtime_voice_agent.demo import (
    DemoAdmissionController,
    DemoRejectionReason,
    DemoReservation,
    keyed_demo_identifier,
)


class FakeClock:
    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _config(**overrides: object) -> DemoRuntimeConfig:
    config = DemoRuntimeConfig(
        enabled=True,
        persona_choices=(),
        max_call_duration_seconds=300.0,
        rate_limit_max_calls=2,
        rate_limit_window_seconds=60.0,
        global_concurrency_limit=2,
        budget_max_calls=10,
        budget_window_seconds=3_600.0,
        reservation_ttl_seconds=10.0,
        persist_transcripts=False,
    )
    return replace(config, **overrides)


@pytest.mark.asyncio
async def test_per_caller_rate_limit_uses_a_rolling_window() -> None:
    clock = FakeClock()
    controller = DemoAdmissionController(_config(), clock=clock)

    assert await controller.admit_entry(caller_key="caller", call_key="call-1") is None
    assert await controller.admit_entry(caller_key="caller", call_key="call-2") is None
    assert (
        await controller.admit_entry(caller_key="caller", call_key="call-3")
        is DemoRejectionReason.RATE_LIMIT
    )

    clock.advance(61)

    assert await controller.admit_entry(caller_key="caller", call_key="call-3") is None


@pytest.mark.asyncio
async def test_pending_reservation_enforces_capacity_and_expires() -> None:
    clock = FakeClock()
    controller = DemoAdmissionController(
        _config(global_concurrency_limit=1),
        clock=clock,
        token_factory=lambda: "reservation-1",
    )
    assert await controller.admit_entry(caller_key="caller-1", call_key="call-1") is None
    reservation = await controller.reserve(
        call_key="call-1",
        persona_id="travel-concierge",
        persona_version=3,
    )
    assert isinstance(reservation, DemoReservation)
    assert (
        await controller.admit_entry(caller_key="caller-2", call_key="call-2")
        is DemoRejectionReason.CAPACITY
    )

    clock.advance(11)

    assert await controller.admit_entry(caller_key="caller-2", call_key="call-2") is None


@pytest.mark.asyncio
async def test_active_lease_releases_capacity_but_not_rolling_budget() -> None:
    clock = FakeClock()
    tokens = iter(("reservation-1", "reservation-2"))
    controller = DemoAdmissionController(
        _config(global_concurrency_limit=1, budget_max_calls=1, budget_window_seconds=30.0),
        clock=clock,
        token_factory=lambda: next(tokens),
    )
    assert await controller.admit_entry(caller_key="caller-1", call_key="call-1") is None
    reservation = await controller.reserve(
        call_key="call-1",
        persona_id="care-coordinator",
        persona_version=1,
    )
    assert isinstance(reservation, DemoReservation)
    lease = await controller.claim(reservation.token)
    assert lease is not None
    await controller.release(lease)

    assert (
        await controller.admit_entry(caller_key="caller-2", call_key="call-2")
        is DemoRejectionReason.BUDGET
    )

    clock.advance(31)

    assert await controller.admit_entry(caller_key="caller-2", call_key="call-2") is None


@pytest.mark.asyncio
async def test_reservation_is_authorized_idempotent_and_single_use() -> None:
    controller = DemoAdmissionController(_config(), token_factory=lambda: "reservation")
    assert (
        await controller.reserve(
            call_key="unknown-call",
            persona_id="history-guide",
            persona_version=4,
        )
        is DemoRejectionReason.INVALID_ENTRY
    )
    assert await controller.admit_entry(caller_key="caller", call_key="call") is None

    first = await controller.reserve(
        call_key="call",
        persona_id="history-guide",
        persona_version=4,
    )
    second = await controller.reserve(
        call_key="call",
        persona_id="history-guide",
        persona_version=4,
    )

    assert isinstance(first, DemoReservation)
    assert second == first
    lease = await controller.claim(first.token)
    assert lease is not None
    assert lease.persona_id == "history-guide"
    assert lease.persona_version == 4
    assert await controller.claim(first.token) is None
    await controller.release(lease)
    await controller.release(lease)


def test_caller_keys_are_keyed_domain_separated_and_do_not_retain_phone_number() -> None:
    phone_number = "+15555550100"
    caller_key = keyed_demo_identifier(
        value=phone_number,
        secret="safe-test-secret",
        domain="caller",
    )

    assert phone_number not in caller_key
    assert caller_key == keyed_demo_identifier(
        value=phone_number,
        secret="safe-test-secret",
        domain="caller",
    )
    assert caller_key != keyed_demo_identifier(
        value=phone_number,
        secret="safe-test-secret",
        domain="call",
    )
