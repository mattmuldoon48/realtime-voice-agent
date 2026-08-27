"""Local-only persona administration, table bootstrap, and transcript retrieval CLI."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Callable, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from realtime_voice_agent.config import (
    AppSettings,
    ConfigurationError,
    PersistenceRuntimeConfig,
    ReadinessRuntimeConfig,
)
from realtime_voice_agent.persistence.dynamodb import DynamoPersistenceStore
from realtime_voice_agent.persistence.errors import PersistenceError
from realtime_voice_agent.persistence.models import Persona, TranscriptView
from realtime_voice_agent.persistence.ports import PersonaRepository, SessionRepository
from realtime_voice_agent.readiness import (
    ReadinessChecker,
    create_readiness_service,
)


class AdminStore(PersonaRepository, SessionRepository, Protocol):
    """Combined repository operations used only by the local CLI."""

    def ensure_tables(self) -> None: ...


StoreFactory = Callable[[PersistenceRuntimeConfig], AdminStore]
ReadinessFactory = Callable[
    [ReadinessRuntimeConfig, PersistenceRuntimeConfig],
    ReadinessChecker,
]


class PersonaDefinition(BaseModel):
    """One domain persona loaded from a configuration document."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    persona_id: str
    name: str
    system_prompt: str
    voice_id: str

    @field_validator("persona_id", "name", "system_prompt", "voice_id")
    @classmethod
    def values_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("persona values must not be blank")
        return value


class PersonaCatalog(BaseModel):
    """Validated collection of independently configurable personas."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    personas: tuple[PersonaDefinition, ...] = Field(min_length=1)

    @field_validator("personas")
    @classmethod
    def persona_ids_must_be_unique(
        cls,
        personas: tuple[PersonaDefinition, ...],
    ) -> tuple[PersonaDefinition, ...]:
        persona_ids = [persona.persona_id for persona in personas]
        if len(persona_ids) != len(set(persona_ids)):
            raise ValueError("persona_id values must be unique")
        return personas


def run(
    argv: Sequence[str] | None = None,
    *,
    store_factory: StoreFactory = DynamoPersistenceStore,
    readiness_factory: ReadinessFactory = create_readiness_service,
) -> int:
    """Run one local administration command and return a process status."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        settings = AppSettings()
        config = settings.to_persistence_runtime_config()
        if args.resource == "preflight":
            checker = readiness_factory(
                settings.to_readiness_runtime_config(),
                config,
            )
            result = asyncio.run(checker.check())
            print(json.dumps(result.to_document(), sort_keys=True))
            return 0 if result.is_ready else 1
        store = store_factory(config)
        output = _dispatch(args, store)
    except (
        ConfigurationError,
        OSError,
        PersistenceError,
        ValidationError,
        ValueError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    if output is not None:
        print(json.dumps(output, indent=2, sort_keys=True))
    return 0


def main() -> None:
    """CLI process entrypoint."""
    raise SystemExit(run())


def _dispatch(args: argparse.Namespace, store: AdminStore) -> object | None:
    if args.resource == "tables":
        store.ensure_tables()
        return {"status": "ok"}
    if args.resource == "personas":
        if args.action == "list":
            return [
                _persona_document(persona, include_prompt=False)
                for persona in store.list_personas()
            ]
        if args.action == "get":
            persona = store.get_persona(args.persona_id)
            if persona is None:
                raise PersistenceError("PERSONA_NOT_FOUND")
            return _persona_document(persona, include_prompt=True)
        if args.action == "put":
            prompt = Path(args.prompt_file).read_text(encoding="utf-8")
            persona = store.put_persona(
                persona_id=args.persona_id,
                name=args.name,
                system_prompt=prompt,
                voice_id=args.voice_id,
                expected_version=args.expected_version,
            )
            return _persona_document(persona, include_prompt=True)
        if args.action == "import":
            catalog = PersonaCatalog.model_validate_json(
                Path(args.file).read_text(encoding="utf-8")
            )
            return _import_personas(store, catalog)
        if args.action == "activate":
            persona = store.activate_persona(
                args.persona_id,
                expected_version=args.expected_version,
            )
            return _persona_document(persona, include_prompt=False)
    if args.resource == "transcripts":
        transcript = store.get_transcript(
            session_id=args.session_id,
            call_sid=args.call_sid,
        )
        return _transcript_document(transcript)
    raise ValueError("unsupported administration command")


def _persona_document(persona: Persona, *, include_prompt: bool) -> dict[str, object]:
    document: dict[str, object] = {
        "persona_id": persona.persona_id,
        "name": persona.name,
        "voice_id": persona.voice_id,
        "version": persona.version,
        "active": persona.active,
        "created_at": persona.created_at.isoformat(),
        "updated_at": persona.updated_at.isoformat(),
    }
    if include_prompt:
        document["system_prompt"] = persona.system_prompt
    return document


def _import_personas(store: AdminStore, catalog: PersonaCatalog) -> dict[str, object]:
    imported: list[dict[str, object]] = []
    unchanged: list[str] = []
    for definition in catalog.personas:
        current = store.get_persona(definition.persona_id)
        if current is not None and _matches_definition(current, definition):
            unchanged.append(definition.persona_id)
            continue
        persona = store.put_persona(
            persona_id=definition.persona_id,
            name=definition.name,
            system_prompt=definition.system_prompt,
            voice_id=definition.voice_id,
            expected_version=current.version if current is not None else 0,
        )
        imported.append(_persona_document(persona, include_prompt=False))
    return {"imported": imported, "unchanged": unchanged}


def _matches_definition(persona: Persona, definition: PersonaDefinition) -> bool:
    return (
        persona.name == definition.name
        and persona.system_prompt == definition.system_prompt
        and persona.voice_id == definition.voice_id
    )


def _transcript_document(transcript: TranscriptView) -> dict[str, object]:
    return {
        "session": transcript.session,
        "turns": [
            {
                **asdict(turn),
                "speaker": turn.speaker.value,
                "created_at": turn.created_at.isoformat(),
            }
            for turn in transcript.turns
        ],
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="realtime-voice-admin")
    resources = parser.add_subparsers(dest="resource", required=True)
    resources.add_parser(
        "preflight",
        help="check read-only AWS and DynamoDB dependency readiness",
    )

    tables = resources.add_parser("tables", help="DynamoDB table operations")
    table_actions = tables.add_subparsers(dest="action", required=True)
    table_actions.add_parser("ensure", help="create missing tables and enable TTL")

    personas = resources.add_parser("personas", help="persona administration")
    persona_actions = personas.add_subparsers(dest="action", required=True)
    persona_actions.add_parser("list", help="list persona summaries")
    get_persona = persona_actions.add_parser("get", help="get one persona")
    get_persona.add_argument("persona_id")
    put_persona = persona_actions.add_parser("put", help="create or update one persona")
    put_persona.add_argument("persona_id")
    put_persona.add_argument("--name", required=True)
    put_persona.add_argument("--voice-id", required=True)
    put_persona.add_argument("--prompt-file", required=True)
    put_persona.add_argument("--expected-version", type=int, required=True)
    import_personas = persona_actions.add_parser(
        "import",
        help="create or update personas from a JSON catalog",
    )
    import_personas.add_argument("--file", required=True)
    activate = persona_actions.add_parser("activate", help="activate one persona")
    activate.add_argument("persona_id")
    activate.add_argument("--expected-version", type=int, required=True)

    transcripts = resources.add_parser("transcripts", help="final transcript retrieval")
    transcript_actions = transcripts.add_subparsers(dest="action", required=True)
    get_transcript = transcript_actions.add_parser("get", help="retrieve one transcript")
    identity = get_transcript.add_mutually_exclusive_group(required=True)
    identity.add_argument("--session-id")
    identity.add_argument("--call-sid")
    return parser


if __name__ == "__main__":
    main()
