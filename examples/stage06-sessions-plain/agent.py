"""
Session, state, and events (Stage 6 -- docs/stage-06-sessions-state.md),
built EXACTLY as that doc's brief -- Event/Session dataclasses and a
session service -- with no framework at all. No LangGraph, no
StateGraph, no checkpointer library. This is the doc's own pseudocode,
made real and runnable.

See examples/stage06-sessions-langgraph/ for the same researcher/writer
handoff built on LangGraph's real primitives instead -- the two examples
are deliberately structured to run the identical scenario, so reading
them side by side shows exactly what a framework buys you (or doesn't)
over the from-scratch version here.

Ollama only -- same LOCAL_MODEL as every other stage03+ example.

The three concepts, as actual code, not pseudocode:

  Event    -- one thing that happened. type + data + timestamp.
              Append-only: __post_init__ aside, nothing in this file
              ever mutates an Event after it's created.
  Session  -- id + state (a mutable dict, "what do we know right now")
              + events (a list[Event], "what happened, in order").
              state is NEVER touched directly outside update_context() --
              every state change is routed through one method so the
              event log stays the authoritative record of why state
              looks the way it does.
  SessionService -- create/get/delete by session id. This file's version
              is PERSISTENT (JSON on disk, one file per session, in
              .sessions/), not the doc's in-memory-only version --
              matching stage06-sessions-langgraph's SqliteSaver bar:
              --history/--list need to work as genuinely separate
              process runs, not just within one Python process. The
              INTERFACE (create/get/delete) is identical to the doc's
              InMemorySessionService; only the storage backend changed,
              which is precisely the point the doc's own docstring makes
              ("swap this for Redis/Postgres later -- the interface
              stays the same either way").

The researcher/writer handoff below is the doc's own researcher_turn/
writer_turn example, verbatim in spirit: writer_turn never calls
researcher_turn. It only reads session.state["research_done"] -- a value
researcher_turn wrote to the SAME session object. Nothing enforces this
separation the way LangGraph's node-input model does (see
stage06-sessions-langgraph's README for that contrast) -- here it's
purely a discipline the two functions follow, which is itself worth
noticing: LangGraph doesn't add a NEW capability, it makes a good habit
structurally unavoidable.

Usage:
    ollama pull llama3.2:3b   # one-time, ~2GB, shared with other examples
    uv run agent.py "the history of Indian spice traders"
        -> prints a session id at the end; every session gets a new one
    uv run agent.py --history <session_id>
        -> re-open the SAME session in a NEW process and print its full
           state + event log, proving it was written to disk, not held
           in memory
    uv run agent.py --list
        -> list every session id ever created under .sessions/
    uv run agent.py --resume <session_id> "a follow-up topic"
        -> REUSE an existing session instead of creating a new one: loads
           its saved state + event history, then appends another
           researcher/writer turn onto it. The session's id and every
           earlier event stay exactly as they were; only state moves
           forward (new topic, new research_notes, new draft).
"""

import json
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

import ollama
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

LOCAL_MODEL = "llama3.2:3b"
SESSIONS_DIR = Path(__file__).parent / ".sessions"


# --- Stage 6's brief, verbatim in shape ------------------------------------

@dataclass
class Event:
    type: str          # "user_message" | "agent_message" | "state_updated" | ...
    data: dict
    timestamp: float = field(default_factory=time.time)


@dataclass
class Session:
    id: str
    state: dict = field(default_factory=dict)
    events: list[Event] = field(default_factory=list)

    def emit(self, event_type: str, data: dict) -> None:
        """Record what happened -- append-only, never mutate a past event."""
        self.events.append(Event(type=event_type, data=data))

    def update_context(self, **changes) -> None:
        """Mutate the current-state snapshot AND log that it changed.
        Never mutate self.state directly elsewhere in the codebase --
        route every change through here so the event log stays authoritative.
        """
        self.state.update(changes)
        self.emit("state_updated", changes)


class PersistentSessionService:
    """Same create/get/delete interface as the doc's InMemorySessionService
    -- the only difference is where a Session actually lives. Each session
    is one JSON file under SESSIONS_DIR, named by its id, so a session
    created in one `uv run` invocation is readable from a completely
    separate one -- the same bar stage06-sessions-langgraph's SqliteSaver
    sets, met here with nothing but the standard library's json module.
    """

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
        """Not in the doc's original interface -- added because a
        persistent service needs an explicit write-back point. The
        in-memory version didn't need this: mutating session.state
        mutated the ONE copy that lived in InMemorySessionService's dict.
        A file-backed session has no such shared copy, so every state
        change needs to be flushed to disk explicitly.
        """
        self._save(session)

    def _save(self, session: Session) -> None:
        self._path(session.id).write_text(json.dumps(asdict(session), indent=2))

    def delete(self, session_id: str) -> None:
        self._path(session_id).unlink(missing_ok=True)

    def list_ids(self) -> list[str]:
        return sorted(p.stem for p in self.sessions_dir.glob("*.json"))


# --- The researcher/writer handoff, the doc's own example -----------------

def researcher_turn(session: Session, topic: str) -> None:
    print(f"  [researcher] researching {topic!r}", file=sys.stderr)
    response = ollama.chat(
        model=LOCAL_MODEL,
        messages=[{
            "role": "user",
            "content": (
                f"Write 2-3 short bullet-point research notes about: {topic}. "
                "Keep it brief -- this is a research summary, not a full report."
            ),
        }],
        think=False,
    )
    session.emit("agent_message", {"from": "researcher", "text": "Research complete."})
    session.update_context(research_notes=response.message.content, research_done=True)


def writer_turn(session: Session) -> None:
    # This is the entire point of the example: writer_turn does NOT call
    # researcher_turn. It reads session.state["research_done"] -- a value
    # the researcher wrote to the SAME session -- and only proceeds
    # because it can see that flag there.
    if not session.state.get("research_done"):
        print("  [writer] research_done is False -- waiting, not drafting", file=sys.stderr)
        return

    print("  [writer] drafting from researcher's notes", file=sys.stderr)
    response = ollama.chat(
        model=LOCAL_MODEL,
        messages=[{
            "role": "user",
            "content": f"Using these research notes, write a 2-sentence summary:\n\n{session.state['research_notes']}",
        }],
        think=False,
    )
    session.emit("agent_message", {"from": "writer", "text": "Draft complete."})
    session.update_context(draft=response.message.content, step="done")


def run_session(topic: str, session_id: str | None = None) -> None:
    """session_id=None creates a brand-new session (the default path).
    session_id=<existing id> REUSES it instead: service.get() loads the
    session's current state + full event history from disk, and this
    function appends a new user_message/researcher/writer turn onto that
    SAME session -- the events already there are never touched, only
    added to, and state fields from the earlier run (e.g. an earlier
    topic's research_notes) are simply overwritten by update_context()
    the same way any state field is. This is what "resuming a session"
    actually means: the session's identity and history persist, only
    the state's snapshot moves forward.
    """
    print(f"  [model] local via Ollama: {LOCAL_MODEL}", file=sys.stderr)
    service = PersistentSessionService()

    if session_id:
        session = service.get(session_id)
        if session is None:
            print(f"No session found for id={session_id} -- can't resume.")
            return
        print(f"  [session] resuming id={session.id} ({len(session.events)} event(s) so far)", file=sys.stderr)
    else:
        session = service.create()
        print(f"  [session] new id={session.id}", file=sys.stderr)

    session.emit("user_message", {"text": topic})
    session.update_context(topic=topic, step="start")
    service.save(session)   # flush after every state change -- see save()'s docstring

    researcher_turn(session, topic)
    service.save(session)

    # A supervisor just runs turns against the shared session until state
    # says done -- same shape as the doc's own usage example, just with
    # explicit save() calls where the doc's in-memory version needed none.
    writer_turn(session)
    service.save(session)

    print(f"\n{session.state.get('draft', '(no draft -- writer never ran)')}")
    print(f"\n(session id: {session.id})")
    print(f"Inspect it later, in a NEW process: uv run agent.py --history {session.id}")
    print(f"Add another turn to THIS session: uv run agent.py --resume {session.id} \"a follow-up topic\"")


def print_history(session_id: str) -> None:
    """Re-open a session by id in a fresh process -- proves state
    genuinely persisted to .sessions/, not just stayed in a variable.
    """
    service = PersistentSessionService()
    session = service.get(session_id)
    if session is None:
        print(f"No session found for id={session_id}.")
        return

    print(f"Final state for session {session_id}:")
    for key, value in session.state.items():
        print(f"  {key}: {value!r}")
    print("\nEvent log (append-only, in order):")
    for i, event in enumerate(session.events):
        print(f"  [{i}] {event.type}: {event.data}")


def list_sessions() -> None:
    service = PersistentSessionService()
    ids = service.list_ids()
    if not ids:
        print("No sessions recorded yet.")
        return
    print(f"{len(ids)} session(s) recorded in {SESSIONS_DIR.name}/:")
    for session_id in ids:
        print(f"  {session_id}")


if __name__ == "__main__":
    argv = sys.argv[1:]
    if argv and argv[0] == "--history":
        print_history(argv[1] if len(argv) > 1 else "")
    elif argv and argv[0] == "--list":
        list_sessions()
    elif argv and argv[0] == "--resume":
        session_id = argv[1] if len(argv) > 1 else ""
        topic = " ".join(argv[2:]) or "the history of Indian spice traders"
        run_session(topic, session_id=session_id)
    else:
        topic = " ".join(argv) or "the history of Indian spice traders"
        run_session(topic)
