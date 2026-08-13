"""
The exact Session/Event/PersistentSessionService pattern from
examples/stage06-sessions-plain/agent.py (Stage 6 --
docs/stage-06-sessions-state.md), reused here so every question asked
against the PDF is recorded as a turn in a resumable, persisted research
session -- not copy-pasted by accident, copied on purpose so this file
has no import dependency on a DIFFERENT example's directory (each
examples/* folder in this repo is its own standalone uv project).

See stage06-sessions-plain/README.md for the full explanation of why
events are append-only, why update_context() is the only path allowed to
change state, and why persistence needs an explicit save() the in-memory
version doesn't. This file only trims the docstrings for brevity; the
mechanics are identical.
"""

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

SESSIONS_DIR = Path(__file__).parent / ".sessions"


@dataclass
class Event:
    type: str
    data: dict
    timestamp: float = field(default_factory=time.time)


@dataclass
class Session:
    id: str
    state: dict = field(default_factory=dict)
    events: list[Event] = field(default_factory=list)

    def emit(self, event_type: str, data: dict) -> None:
        self.events.append(Event(type=event_type, data=data))

    def update_context(self, **changes) -> None:
        self.state.update(changes)
        self.emit("state_updated", changes)


class PersistentSessionService:
    def __init__(self, sessions_dir: Path = SESSIONS_DIR):
        self.sessions_dir = sessions_dir
        self.sessions_dir.mkdir(exist_ok=True)

    def _path(self, session_id: str) -> Path:
        return self.sessions_dir / f"{session_id}.json"

    def create(self) -> Session:
        session = Session(id=str(uuid.uuid4()))
        self._save(session)
        return session

    def get(self, session_id: str) -> Session | None:
        path = self._path(session_id)
        if not path.exists():
            return None
        raw = json.loads(path.read_text())
        events = [Event(**e) for e in raw["events"]]
        return Session(id=raw["id"], state=raw["state"], events=events)

    def save(self, session: Session) -> None:
        self._save(session)

    def _save(self, session: Session) -> None:
        self._path(session.id).write_text(json.dumps(asdict(session), indent=2))

    def delete(self, session_id: str) -> None:
        self._path(session_id).unlink(missing_ok=True)

    def list_ids(self) -> list[str]:
        return sorted(p.stem for p in self.sessions_dir.glob("*.json"))
