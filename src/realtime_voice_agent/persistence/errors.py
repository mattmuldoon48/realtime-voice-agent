"""Controlled persistence failures safe for operator-facing commands."""

from __future__ import annotations


class PersistenceError(RuntimeError):
    """Normalized DynamoDB failure with an explicit safe retry decision."""

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable


class PersistenceConflictError(PersistenceError):
    """An optimistic version or write-once condition failed."""


class PersistenceNotFoundError(PersistenceError):
    """A requested persona, session, or transcript was not found."""
