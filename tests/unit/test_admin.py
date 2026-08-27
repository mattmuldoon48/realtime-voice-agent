"""Unit tests for local persona administration and transcript retrieval commands."""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import pytest

from realtime_voice_agent.admin.cli import run
from realtime_voice_agent.persistence.models import (
    Persona,
    SessionStart,
    SessionTerminal,
    TranscriptTurn,
    TranscriptView,
)
from realtime_voice_agent.readiness import ReadinessErrorCode, ReadinessResult
from realtime_voice_agent.transcript import TranscriptSpeaker


@pytest.fixture(autouse=True)
def _configure_aws_region(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AWS_REGION", "us-east-1")


class FakeAdminStore:
    def __init__(self) -> None:
        now = datetime(2026, 8, 4, tzinfo=UTC)
        self.persona = Persona(
            persona_id="concierge",
            name="Concierge",
            system_prompt="Answer briefly.",
            voice_id="matthew",
            version=1,
            active=False,
            created_at=now,
            updated_at=now,
        )
        self.put_values: dict[str, object] | None = None

    def ensure_tables(self) -> None:
        return None

    def list_personas(self) -> Sequence[Persona]:
        return (self.persona,)

    def get_persona(self, persona_id: str) -> Persona | None:
        return self.persona if persona_id == self.persona.persona_id else None

    def get_active_persona(self) -> Persona:
        return self.persona

    def put_persona(
        self,
        *,
        persona_id: str,
        name: str,
        system_prompt: str,
        voice_id: str,
        expected_version: int,
    ) -> Persona:
        self.put_values = {
            "persona_id": persona_id,
            "name": name,
            "system_prompt": system_prompt,
            "voice_id": voice_id,
            "expected_version": expected_version,
        }
        return self.persona

    def activate_persona(self, persona_id: str, *, expected_version: int) -> Persona:
        return self.persona

    def create_session(self, session: SessionStart) -> None:
        raise NotImplementedError

    def mark_session_active(self, session_id: str, activated_at: str) -> None:
        raise NotImplementedError

    def append_transcript_turn(self, turn: TranscriptTurn) -> bool:
        raise NotImplementedError

    def finish_session(self, terminal: SessionTerminal) -> bool:
        raise NotImplementedError

    def get_transcript(
        self,
        *,
        session_id: str | None = None,
        call_sid: str | None = None,
    ) -> TranscriptView:
        identity = session_id or call_sid
        return TranscriptView(
            session={"session_id": "session-1", "lookup": identity or ""},
            turns=(
                TranscriptTurn(
                    session_id="session-1",
                    turn_number=1,
                    speaker=TranscriptSpeaker.CALLER,
                    text="Hello",
                    source_event_id="final-1",
                    created_at=datetime(2026, 8, 4, tzinfo=UTC),
                    expires_at=1_786_435_200,
                ),
            ),
        )


class CatalogAdminStore(FakeAdminStore):
    def __init__(self) -> None:
        super().__init__()
        self.personas: dict[str, Persona] = {}
        self.put_count = 0

    def list_personas(self) -> Sequence[Persona]:
        return tuple(self.personas[persona_id] for persona_id in sorted(self.personas))

    def get_persona(self, persona_id: str) -> Persona | None:
        return self.personas.get(persona_id)

    def put_persona(
        self,
        *,
        persona_id: str,
        name: str,
        system_prompt: str,
        voice_id: str,
        expected_version: int,
    ) -> Persona:
        current = self.personas.get(persona_id)
        assert expected_version == (current.version if current is not None else 0)
        now = datetime(2026, 8, 4, tzinfo=UTC)
        persona = Persona(
            persona_id=persona_id,
            name=name,
            system_prompt=system_prompt,
            voice_id=voice_id,
            version=expected_version + 1,
            active=current.active if current is not None else False,
            created_at=current.created_at if current is not None else now,
            updated_at=now,
        )
        self.personas[persona_id] = persona
        self.put_count += 1
        return persona


class FakeReadiness:
    def __init__(self, result: ReadinessResult) -> None:
        self._result = result

    async def check(self) -> ReadinessResult:
        return self._result


def test_persona_put_reads_prompt_file_and_forwards_expected_version(
    tmp_path: Path,
) -> None:
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("Use the configured behavior.\n", encoding="utf-8")
    store = FakeAdminStore()

    status = run(
        [
            "personas",
            "put",
            "concierge",
            "--name",
            "Concierge",
            "--voice-id",
            "matthew",
            "--prompt-file",
            str(prompt_file),
            "--expected-version",
            "0",
        ],
        store_factory=lambda _config: store,
    )

    assert status == 0
    assert store.put_values == {
        "persona_id": "concierge",
        "name": "Concierge",
        "system_prompt": "Use the configured behavior.\n",
        "voice_id": "matthew",
        "expected_version": 0,
    }


def test_persona_import_loads_domain_catalog_and_is_idempotent(
    capsys: pytest.CaptureFixture[str],
) -> None:
    store = CatalogAdminStore()
    command = ["personas", "import", "--file", "config/personas.example.json"]

    first_status = run(command, store_factory=lambda _config: store)

    assert first_status == 0
    first_document = json.loads(capsys.readouterr().out)
    expected_ids = {
        "care-coordinator",
        "financial-services-assistant",
        "history-guide",
        "travel-concierge",
    }
    assert {persona["persona_id"] for persona in first_document["imported"]} == expected_ids
    assert first_document["unchanged"] == []
    assert {persona.name for persona in store.personas.values()} == {
        "Care Coordinator",
        "Financial Services Assistant",
        "History Guide",
        "Travel Concierge",
    }
    assert store.put_count == 4

    second_status = run(command, store_factory=lambda _config: store)

    assert second_status == 0
    second_document = json.loads(capsys.readouterr().out)
    assert second_document["imported"] == []
    assert set(second_document["unchanged"]) == expected_ids
    assert store.put_count == 4


def test_transcript_command_emits_readable_numeric_ordered_json(
    capsys: pytest.CaptureFixture[str],
) -> None:
    store = FakeAdminStore()

    status = run(
        ["transcripts", "get", "--session-id", "session-1"],
        store_factory=lambda _config: store,
    )

    assert status == 0
    captured = capsys.readouterr()
    document = json.loads(captured.out)
    assert document["session"]["session_id"] == "session-1"
    assert document["turns"] == [
        {
            "created_at": "2026-08-04T00:00:00+00:00",
            "expires_at": 1_786_435_200,
            "session_id": "session-1",
            "source_event_id": "final-1",
            "speaker": "CALLER",
            "text": "Hello",
            "turn_number": 1,
        }
    ]


@pytest.mark.parametrize(
    ("result", "expected_status", "expected_document"),
    [
        (ReadinessResult.ready(), 0, {"status": "ready"}),
        (
            ReadinessResult.unavailable(ReadinessErrorCode.TABLE_MISSING),
            1,
            {"status": "not_ready", "error_code": "TABLE_MISSING"},
        ),
    ],
)
def test_preflight_uses_shared_readiness_and_returns_process_status(
    result: ReadinessResult,
    expected_status: int,
    expected_document: dict[str, str],
    capsys: pytest.CaptureFixture[str],
) -> None:
    status = run(
        ["preflight"],
        store_factory=lambda _config: pytest.fail("preflight constructed runtime store"),
        readiness_factory=lambda _readiness, _persistence: FakeReadiness(result),
    )

    assert status == expected_status
    assert json.loads(capsys.readouterr().out) == expected_document
