"""In-process public-demo admission, rate, capacity, and budget controls."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import secrets
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

from realtime_voice_agent.config import DemoRuntimeConfig

type Clock = Callable[[], float]
type TokenFactory = Callable[[], str]


class DemoRejectionReason(StrEnum):
    """Bounded public reasons that never expose internal capacity details."""

    RATE_LIMIT = "RATE_LIMIT"
    CAPACITY = "CAPACITY"
    BUDGET = "BUDGET"
    INVALID_ENTRY = "INVALID_ENTRY"


@dataclass(frozen=True, slots=True)
class DemoReservation:
    """Short-lived authorization passed to Twilio as an opaque parameter."""

    token: str
    persona_id: str
    persona_version: int


@dataclass(frozen=True, slots=True)
class DemoCallLease:
    """One claimed global demo slot released when the media session closes."""

    token: str
    persona_id: str
    persona_version: int


@dataclass(frozen=True, slots=True)
class _PendingReservation:
    call_key: str
    persona_id: str
    persona_version: int
    expires_at: float


class DemoAdmissionController:
    """Serialize bounded admission state for the single-process demo runtime."""

    def __init__(
        self,
        config: DemoRuntimeConfig,
        *,
        clock: Clock | None = None,
        token_factory: TokenFactory | None = None,
    ) -> None:
        self._config = config
        self._clock = clock or time.monotonic
        self._token_factory = token_factory or (lambda: secrets.token_urlsafe(24))
        self._lock = asyncio.Lock()
        self._caller_attempts: dict[str, deque[float]] = {}
        self._budget_starts: deque[float] = deque()
        self._menu_entries: dict[str, float] = {}
        self._call_reservations: dict[str, str] = {}
        self._pending: dict[str, _PendingReservation] = {}
        self._active: dict[str, DemoCallLease] = {}

    async def admit_entry(
        self,
        *,
        caller_key: str,
        call_key: str,
    ) -> DemoRejectionReason | None:
        """Rate-limit one signed inbound call and authorize its menu callback."""
        async with self._lock:
            now = self._clock()
            self._cleanup(now)
            if call_key in self._menu_entries or call_key in self._call_reservations:
                return None
            attempts = self._caller_attempts.setdefault(caller_key, deque())
            if len(attempts) >= self._config.rate_limit_max_calls:
                return DemoRejectionReason.RATE_LIMIT
            attempts.append(now)
            rejection = self._availability_rejection()
            if rejection is not None:
                return rejection
            self._menu_entries[call_key] = now + self._config.reservation_ttl_seconds
            return None

    async def reserve(
        self,
        *,
        call_key: str,
        persona_id: str,
        persona_version: int,
    ) -> DemoReservation | DemoRejectionReason:
        """Reserve capacity and budget exactly once for a validated menu selection."""
        async with self._lock:
            now = self._clock()
            self._cleanup(now)
            existing_token = self._call_reservations.get(call_key)
            if existing_token is not None:
                existing = self._pending.get(existing_token)
                if (
                    existing is not None
                    and existing.persona_id == persona_id
                    and existing.persona_version == persona_version
                ):
                    return DemoReservation(
                        token=existing_token,
                        persona_id=persona_id,
                        persona_version=persona_version,
                    )
                return DemoRejectionReason.INVALID_ENTRY
            if call_key not in self._menu_entries:
                return DemoRejectionReason.INVALID_ENTRY
            rejection = self._availability_rejection()
            if rejection is not None:
                self._menu_entries.pop(call_key, None)
                return rejection
            token = self._new_unique_token()
            self._pending[token] = _PendingReservation(
                call_key=call_key,
                persona_id=persona_id,
                persona_version=persona_version,
                expires_at=now + self._config.reservation_ttl_seconds,
            )
            self._call_reservations[call_key] = token
            self._menu_entries.pop(call_key, None)
            self._budget_starts.append(now)
            return DemoReservation(
                token=token,
                persona_id=persona_id,
                persona_version=persona_version,
            )

    async def claim(self, token: str) -> DemoCallLease | None:
        """Convert one unexpired reservation into an active call lease."""
        async with self._lock:
            now = self._clock()
            self._cleanup(now)
            pending = self._pending.pop(token, None)
            if pending is None:
                return None
            self._call_reservations.pop(pending.call_key, None)
            lease = DemoCallLease(
                token=token,
                persona_id=pending.persona_id,
                persona_version=pending.persona_version,
            )
            self._active[token] = lease
            return lease

    async def release(self, lease: DemoCallLease) -> None:
        """Release one active slot idempotently."""
        async with self._lock:
            self._active.pop(lease.token, None)

    def _availability_rejection(self) -> DemoRejectionReason | None:
        if len(self._pending) + len(self._active) >= self._config.global_concurrency_limit:
            return DemoRejectionReason.CAPACITY
        if len(self._budget_starts) >= self._config.budget_max_calls:
            return DemoRejectionReason.BUDGET
        return None

    def _cleanup(self, now: float) -> None:
        rate_cutoff = now - self._config.rate_limit_window_seconds
        empty_callers: list[str] = []
        for caller_key, attempts in self._caller_attempts.items():
            while attempts and attempts[0] <= rate_cutoff:
                attempts.popleft()
            if not attempts:
                empty_callers.append(caller_key)
        for caller_key in empty_callers:
            self._caller_attempts.pop(caller_key, None)

        budget_cutoff = now - self._config.budget_window_seconds
        while self._budget_starts and self._budget_starts[0] <= budget_cutoff:
            self._budget_starts.popleft()

        expired_entries = [
            call_key for call_key, expires_at in self._menu_entries.items() if expires_at <= now
        ]
        for call_key in expired_entries:
            self._menu_entries.pop(call_key, None)

        expired_tokens = [
            token for token, reservation in self._pending.items() if reservation.expires_at <= now
        ]
        for token in expired_tokens:
            reservation = self._pending.pop(token)
            self._call_reservations.pop(reservation.call_key, None)

    def _new_unique_token(self) -> str:
        for _ in range(3):
            token = self._token_factory()
            if token and token not in self._pending and token not in self._active:
                return token
        raise RuntimeError("Unable to allocate a unique demo reservation")


def keyed_demo_identifier(*, value: str | None, secret: str, domain: str) -> str:
    """Create an in-memory correlation key without retaining a caller identifier."""
    normalized = value.strip() if value and value.strip() else "anonymous"
    message = f"{domain}:{normalized}".encode()
    return hmac.new(secret.encode(), message, hashlib.sha256).hexdigest()
