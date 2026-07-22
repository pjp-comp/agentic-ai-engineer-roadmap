[← Stage 04](stage-04-memory-state.md) · Stage 05 of 13 · **Next:** [Stage 06 →](stage-06-single-agent.md)

# Stage 05 — Sessions, State + Events

The session object · state vs. events · an in-memory session service · updating context mid-run · inter-agent communication via a shared session

## Why this matters

Stage 4 gave you memory *types* (short-term, persistent, long-term) but never named the container all of them live inside. That container is the **session** — the object that scopes one run of an agent: a stable ID, a mutable state dict, and an ordered log of everything that happened. Every framework (LangGraph, CrewAI, Google's ADK, Claude's own Managed Agents) has some version of this object under a different name, because there's no way to build a real agent without it. Skipping straight to a framework's `Session` class without understanding what it's *for* means you can use it but can't debug it when it misbehaves.

## Session vs. state vs. events — three different things

It's easy to blur these into "the session," but they answer different questions and a bug in one looks completely different from a bug in another:

| Concept | Answers | Example |
|---|---|---|
| **Session** | "Which run is this?" | `session_id="run_8f3a"`, created at 14:02, owner user `u_412` |
| **State** | "What do we currently know?" | `{"user_name": "Alice", "cart_total": 42.50, "step": 3}` — a snapshot, overwritten as things change |
| **Events** | "What happened, in order?" | `[UserMessage("hi"), ToolCall("lookup_price"), ToolResult(...), StateUpdated("cart_total", 42.50)]` — append-only, never overwritten |

**State is derived from events, not the other way around.** If you replay a session's event log from the start, you should be able to reconstruct its current state exactly. This is why events are append-only: they're the source of truth; state is a cache of "where the events left off." When state and the event log disagree, trust the event log and treat the state snapshot as the thing that's wrong.

## Brief — build all three, plus a service to manage many of them

```python
import time
import uuid
from dataclasses import dataclass, field


@dataclass
class Event:
    type: str          # "user_message" | "tool_call" | "tool_result" | "state_updated"
    data: dict
    timestamp: float = field(default_factory=time.time)


@dataclass
class Session:
    id: str
    state: dict = field(default_factory=dict)
    events: list[Event] = field(default_factory=list)

    def emit(self, event_type: str, data: dict):
        """Record what happened — append-only, never mutate a past event."""
        self.events.append(Event(type=event_type, data=data))

    def update_context(self, **changes):
        """Mutate the current-state snapshot AND log that it changed.
        Never mutate self.state directly elsewhere in the codebase —
        route every change through here so the event log stays authoritative."""
        self.state.update(changes)
        self.emit("state_updated", changes)


class InMemorySessionService:
    """The simplest possible session store: a dict, guarded by ID.
    Swap this for Redis/Postgres later (see Stage 4's PersistentState) —
    the interface below is what stays the same either way."""

    def __init__(self):
        self._sessions: dict[str, Session] = {}

    def create(self) -> Session:
        session = Session(id=str(uuid.uuid4()))
        self._sessions[session.id] = session
        return session

    def get(self, session_id: str) -> Session | None:
        return self._sessions.get(session_id)

    def delete(self, session_id: str):
        self._sessions.pop(session_id, None)


# --- Usage: one agent turn ---
service = InMemorySessionService()
session = service.create()

session.emit("user_message", {"text": "What's my order total?"})
session.update_context(step="looking_up_order")   # state changes + it's logged

# ... agent calls a tool ...
session.emit("tool_call", {"name": "get_order_total", "args": {"order_id": "o_9"}})
session.emit("tool_result", {"total": 42.50})
session.update_context(cart_total=42.50, step="answered")

print(session.state)   # {"step": "answered", "cart_total": 42.50}
print(len(session.events))  # full audit trail, in order
```

## Two agents communicating through a shared session

This is the mechanism underneath Stage 7's supervisor pattern, made explicit. Two agents don't call each other directly — they read and write **the same session's state**, and communicate by watching for events the other one emitted. This is what "message passing" and "handoffs" (Stage 7's vocabulary) actually resolve to at the state layer:

```python
def researcher_turn(session: Session):
    session.emit("agent_message", {"from": "researcher", "text": "Found 3 sources."})
    session.update_context(research_done=True, sources=["a", "b", "c"])

def writer_turn(session: Session):
    if not session.state.get("research_done"):
        return  # writer waits — it reads researcher's state, doesn't call researcher directly
    session.emit("agent_message", {"from": "writer", "text": "Drafting from sources."})
    session.update_context(draft="...", step="done")

# A supervisor just runs turns against the shared session until state says done:
session = service.create()
researcher_turn(session)
writer_turn(session)  # only proceeds because it can see research_done=True in shared state
```

Neither agent has a reference to the other — only to the session. This is why it composes: adding a third agent means giving it read/write access to the same session, not wiring a new direct channel to every existing agent.

## Sources

| Type | Resource |
|------|----------|
| Docs | [LangGraph — persistence & checkpointer concepts](https://docs.langchain.com/oss/python/langgraph/persistence) |
| Docs | [Google Agent Development Kit — Sessions, State, Events](https://google.github.io/adk-docs/sessions/) — the clearest existing writeup of this exact session/state/event split |
| Managed Agents | [Claude Platform — Sessions](https://platform.claude.com/docs/en/managed-agents/sessions) — Anthropic's own hosted session lifecycle (`idle` / `running` / `terminated`), for comparison against the in-memory version above |
| Concept | Event sourcing — search "event sourcing pattern" for the older, non-AI-specific origin of "state is derived from an append-only log" |

## Done when

You can explain, without looking it up, why `session.state` should never be mutated directly outside `update_context()` — and a second agent reading the same session can tell what the first one did by reading its `state`, without the two agents ever calling each other's code.

---
[← Stage 04 — Memory + State Management](stage-04-memory-state.md) · [Back to roadmap](../README.md) · **Next:** [Stage 06 — Single-Agent Workflows →](stage-06-single-agent.md)
