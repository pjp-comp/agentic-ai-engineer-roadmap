[← Back to roadmap](../../README.md)

# Stage 6 — Sessions, State + Events, as Real LangGraph Mechanics

[Stage 6](../../docs/stage-06-sessions-state.md) teaches session/state/events with hand-rolled `Session`/`Event` dataclasses and an `InMemorySessionService`. This example builds the same three concepts on LangGraph's actual persistence primitives instead — so you see what a framework gives you for free, and what it doesn't, after already understanding the concepts from scratch.

**Ollama only** — `llama3.2:3b`, no Claude path.

## How LangGraph maps onto Stage 6's three concepts

| Stage 6 concept | Answers | LangGraph equivalent |
|---|---|---|
| **Session** | "Which run is this?" | `thread_id` in the config passed to `app.invoke()` — the checkpointer scopes *all* state to a thread_id, the same way `Session.id` scopes a run |
| **State** | "What do we currently know?" | The `SessionState` TypedDict — every node returns a partial update, LangGraph merges it into current state |
| **Events** | "What happened, in order?" | **Not a first-class LangGraph concept.** `app.get_state_history()` gives a trail of state *snapshots*, not a named, typed event log. This example adds an explicit `events: list[dict]` field to the state schema itself, so "what happened, with a type" stays a visible list — the same thing Stage 6's `Event` class gives you, just modeled as data instead of a framework feature |

That gap in the middle row is deliberately not papered over — it's real, useful information about what LangGraph does and doesn't hand you for free.

## Why SqliteSaver, not InMemorySaver

LangGraph ships an `InMemorySaver` that "persists" state, but only for the lifetime of one Python process — that's not really persistence, it's a variable staying in scope. This example uses `SqliteSaver` instead, writing checkpoints to a real `.sessions.db` file, so `--resume`/`--history` below work as genuinely separate `uv run` invocations — the same bar [Stage 4's `PersistentState`](../../docs/stage-04-memory-state.md) sets for "survives a restart."

## The two-agent handoff

`researcher_node` runs first, writes `research_notes` and sets `research_done: True` into shared state. `writer_node` runs second — and **never calls `researcher_node` directly**. It only checks `state["research_done"]`. This is Stage 6's actual point about inter-agent communication, made structurally unavoidable: a LangGraph node's only inputs are the graph state and its own code, so "agents communicate through shared state, not direct calls" isn't a design choice here — it's the only mechanism available.

## Run it

```bash
cd examples/stage06-sessions-langgraph
ollama pull llama3.2:3b   # one-time, ~2GB, shared with other examples
uv run agent.py "Northwind Traders Q3 results"
```

Expected output — the researcher runs, the writer waits for `research_done` before drafting, and a session ID is printed at the end:

```
  [researcher] researching 'Northwind Traders Q3 results'
  [writer] drafting from researcher's notes

<a two-sentence summary>

(session thread_id: 0481eb29-9592-4879-9702-140cf28f4f6a)
Inspect it later, in a NEW process: uv run agent.py --history 0481eb29-...
```

Now open a **completely new process** and inspect that same session — this is the real persistence test, not just "the variable is still there":

```bash
uv run agent.py --history 0481eb29-9592-4879-9702-140cf28f4f6a   # use your own printed thread_id
```

You should see the full final state and the event log, e.g.:

```
Event log (append-only, in order):
  [0] agent_message: {'from': 'researcher', 'text': 'Research complete.'}
  [1] agent_message: {'from': 'writer', 'text': 'Draft complete.'}
```

Run a second, unrelated session and confirm the two stay isolated:

```bash
uv run agent.py "the benefits of async Python"
uv run agent.py --list   # should show both thread_ids, independently
```

## What to look at closely

- **`_emit()` returns a new list instead of mutating `state["events"]` in place** — LangGraph nodes are supposed to return *updates*, not mutate the state object directly. "Append-only" here means `state["events"] + [new_item]`, not `.append()`. Mutating shared state directly inside a node is a common LangGraph beginner mistake this function's shape avoids by construction.
- **`writer_node`'s early return when `research_done` is `False`** — in this example the graph edges (`researcher` → `writer`, always in that order) mean this branch never actually triggers; it's here because a *real* multi-agent graph often has the writer reachable from more than one path (a retry edge, a parallel branch), where the check genuinely matters. Included so the pattern is visible even though this specific graph's linear edges make it currently unreachable.
- **`list_sessions()` reads the checkpointer's own SQLite table directly** rather than keeping a separate session index — Stage 6's `InMemorySessionService` keeps a dict of sessions for exactly this purpose; `SqliteSaver`'s `checkpoints` table already has the equivalent information, so this reuses it instead of duplicating a second source of truth.
- **Every `run_session()` call gets a brand-new `thread_id`** (`uuid.uuid4()`) — this example always starts a fresh session rather than resuming an existing one on the default path. Real resumption (continuing an existing thread_id's conversation across multiple turns) is a straightforward extension: pass a known `thread_id` into `run_session()` instead of always generating one, the same idea `--history` already demonstrates for *reading* a session, just applied to *continuing* one.

## Where this goes next

This is the mechanism underneath [Stage 8](../../docs/stage-08-multi-agent.md)'s supervisor pattern, made explicit at the state layer — a supervisor coordinating three or more agents is this same shared-state pattern with more nodes and a routing function deciding which one runs next, not a different architecture. See [Stage 8's graph-engineering section](../../docs/stage-08-multi-agent.md#graph-engineering-in-depth--why-the-topology-is-a-first-class-design-decision) for what changes once a graph has enough heterogeneous nodes that the topology itself needs deliberate design.
