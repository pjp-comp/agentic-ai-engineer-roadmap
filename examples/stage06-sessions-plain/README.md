[← Back to roadmap](../../README.md)

# Stage 6 — Sessions, State + Events, With No Framework At All

[Stage 6](../../docs/stage-06-sessions-state.md)'s brief is a hand-rolled `Event`/`Session`/`InMemorySessionService` — three plain classes, no library. This example builds exactly that, made persistent and runnable, with the same researcher/writer handoff as [`stage06-sessions-langgraph`](../stage06-sessions-langgraph/) so the two examples can be read side by side: same scenario, same CLI shape, one built on LangGraph's primitives, one built on nothing but the standard library.

**Ollama only** — `llama3.2:3b`, no Claude path. No LangGraph, no framework of any kind.

## How it works

- **`Event`** — `type` + `data` + a `timestamp`. One thing that happened. Nothing in this file ever mutates an `Event` after it's created.
- **`Session`** — an `id`, a mutable `state` dict ("what do we know right now"), and an `events` list ("what happened, in order"). `session.state` is never touched directly — every change goes through `update_context()`, which updates state **and** logs an `state_updated` event in the same call, so the event log always explains *why* state looks the way it does.
- **`PersistentSessionService`** — the doc's `InMemorySessionService`, with the same `create()`/`get()`/`delete()` interface, but backed by one JSON file per session under `.sessions/` instead of an in-memory dict. This is the doc's own advice ("swap this for Redis/Postgres later — the interface stays the same") taken literally: only the storage changed, not the shape callers use.

The researcher/writer handoff is the doc's own example, verbatim in spirit:

```python
def researcher_turn(session, topic):
    ...
    session.update_context(research_notes=..., research_done=True)

def writer_turn(session):
    if not session.state.get("research_done"):
        return  # writer waits -- reads researcher's state, never calls researcher directly
    ...
```

## Run it

```bash
cd examples/stage06-sessions-plain
ollama pull llama3.2:3b   # one-time, ~2GB, shared with other examples
uv run agent.py "the history of Indian spice traders"
```

```
  [researcher] researching 'the history of Indian spice traders'
  [writer] drafting from researcher's notes

Here is a 2-sentence summary:

The Indian spice trade has a rich history dating back to the ancient Indus Valley Civilization, with spices such as cinnamon and pepper being traded along the Silk Road network. The expansion of the Portuguese, Dutch, and British East India Company established new sea routes and trading posts, ultimately leading to the exploitation and colonization of Indian merchants in the 17th and 18th centuries.

(session id: 9e07f286-5127-4efb-9da0-a118c0ed2f18)
Inspect it later, in a NEW process: uv run agent.py --history 9e07f286-...
```

Open a **completely new process** and inspect that same session — the real persistence test:

```bash
uv run agent.py --history 9e07f286-5127-4efb-9da0-a118c0ed2f18   # use your own printed id
```

```
Final state for session 9e07f286-...:
  topic: 'the history of Indian spice traders'
  step: 'done'
  research_notes: "..."
  research_done: True
  draft: "..."

Event log (append-only, in order):
  [0] user_message: {'text': 'the history of Indian spice traders'}
  [1] state_updated: {'topic': 'the history of Indian spice traders', 'step': 'start'}
  [2] agent_message: {'from': 'researcher', 'text': 'Research complete.'}
  [3] state_updated: {'research_notes': '...', 'research_done': True}
  [4] agent_message: {'from': 'writer', 'text': 'Draft complete.'}
  [5] state_updated: {'draft': '...', 'step': 'done'}
```

Run a second, unrelated session and confirm the two stay isolated:

```bash
uv run agent.py "the benefits of async Python"
uv run agent.py --list   # shows both session ids, independently
```

**Reuse an existing session with `--resume`** — instead of creating a new one, load a session by id and append another researcher/writer turn onto it:

```bash
uv run agent.py --resume 9e07f286-5127-4efb-9da0-a118c0ed2f18 "how European colonization changed that trade"
```

```
  [session] resuming id=9e07f286-... (6 event(s) so far)
  [researcher] researching 'how European colonization changed that trade'
  [writer] drafting from researcher's notes

<a new 2-sentence summary, built from THIS turn's research>

(session id: 9e07f286-5127-4efb-9da0-a118c0ed2f18)
```

Same session id, same file on disk — `--history` on it now shows **12** events, not 6: the original turn's events `[0]`–`[5]` untouched, the resumed turn's events `[6]`–`[11]` appended after them. `state["topic"]`, `state["research_notes"]`, and `state["draft"]` all move forward to the new turn's values (state is a snapshot, so it only ever reflects the *latest* write to each key) — but nothing already in `events` is ever rewritten or lost. Resuming a session id that doesn't exist fails cleanly (`No session found for id=... -- can't resume.`) instead of silently creating a new one under a typo'd id.

## What to look at closely

- **`update_context()` is the only path allowed to change `session.state`** — nothing in `researcher_turn`/`writer_turn` writes `session.state[...] = ...` directly. This is enforced by convention here, not by the type system — worth noticing as the actual tradeoff against the LangGraph version, where a node's return value is the *only* way to change state, so the equivalent discipline is structurally unavoidable rather than a rule you have to follow.
- **`service.save(session)` is called after every state-changing step, not once at the end** — a persistent, file-backed session has no single shared copy the way an in-memory dict does, so a change that's never flushed to disk doesn't exist as far as a second process is concerned. The crash-recovery test worth trying: kill `agent.py` mid-run (e.g. `Ctrl-C` right after `[researcher] researching...` prints) and then run `--history` on the printed session id — you'll see state frozen at exactly the last `save()` that completed, with the event log confirming how far it got. This is real, observable evidence of why the save-after-every-step placement matters, not just a docstring claim.
- **`PersistentSessionService.list_ids()` reads the directory itself** (`.sessions/*.json`) rather than keeping a separate index — same principle as [`stage06-sessions-langgraph`](../stage06-sessions-langgraph/)'s `list_sessions()` reading the checkpointer's own SQLite table: the storage already has the answer, so don't duplicate it.
- **No `thread_id`, no checkpointer, no `StateGraph`** — every mechanism here is either a dataclass, a dict, or a JSON file. If a beginner asks "what is LangGraph actually giving me," this file is the honest baseline to compare it against — read this one first, then `stage06-sessions-langgraph`'s README for what changes (mostly: state updates become structurally enforced instead of convention, and persistence becomes one battle-tested library call instead of manual `save()` placement).
- **`run_session(topic, session_id=...)` is the same function for both "start new" and "resume existing"** — the only branch is `service.get()` vs. `service.create()` at the top; everything after that (emit a `user_message`, run the researcher, run the writer, save after each step) is identical code path either way. This is worth noticing: "resuming a session" isn't a special mode requiring different logic, it's just *which* `Session` object the same turns get applied to. `service.get()` reconstructs the full `Session` (state **and** every past `Event`) from its JSON file — resuming isn't replaying the events to rebuild state, it's reading the already-current `state` snapshot straight off disk, exactly as [`_build_vector_store()`'s persisted-index pattern](../stage05-rag-langgraph/) reads an already-built index instead of recomputing it.

## Where this goes next

This is the mechanism underneath [Stage 8](../../docs/stage-08-multi-agent.md)'s supervisor pattern, made as explicit as it gets — a third agent just needs read/write access to the same `Session` object, not a new direct channel to either existing agent. Compare against [`stage06-sessions-langgraph`](../stage06-sessions-langgraph/) to see the identical scenario with a framework underneath it, and against [`stage08-supervisor-langgraph`](../stage08-supervisor-langgraph/) for what a *real* supervisor (a router deciding "who's next," not just two fixed turns in sequence) adds on top of this.
