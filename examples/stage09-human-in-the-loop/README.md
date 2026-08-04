[← Back to roadmap](../../README.md)

# Stage 9 — An Approval Gate, on Real LangGraph `interrupt_before`

[Stage 9](../../docs/stage-09-human-in-the-loop.md)'s brief is four lines: compile a graph with `interrupt_before` on a high-risk node, `update_state()` with a human's decision, then resume with `invoke(None, ...)`. This example builds exactly that — a graph that drafts a refund, pauses *before* sending it, and only executes once a human approves it from a **separate process**.

**Ollama only** — `llama3.2:3b`, no Claude path. `SqliteSaver` stands in for the brief's `PostgresSaver` — same interrupt/resume mechanics, no database server required for a learning example.

## How it works

```
START -> draft_action -> [INTERRUPT] -> execute_high_risk_action -> END
```

- **`draft_action`** — an agent node. The model decides *what* the refund should be (amount + reason) but the graph is compiled with `interrupt_before=["execute_high_risk_action"]`, so nothing happens after this — the graph simply pauses.
- **`execute_high_risk_action`** — the actual side effect (a fake `send_refund`). By the time this code runs, a human has already written `approved: True/False` into the paused graph's state via `update_state()`.

The interrupt sits **before** the side effect, not after — that's the whole point of an approval gate versus an audit log. A human reviews the *proposal*, not a record of something that already happened.

## Run it

```bash
cd examples/stage09-human-in-the-loop
ollama pull llama3.2:3b   # one-time, ~2GB, shared with other examples
uv run agent.py "refund the customer $45 for a damaged item"
```

This drafts the action and pauses — nothing is executed yet:

```
Proposed action: refund $45.00
Reason: The item was received in a damaged condition.

(paused for approval -- thread_id: 6e7a633a-...)
Approve:  uv run agent.py --approve 6e7a633a-...
Reject:   uv run agent.py --reject 6e7a633a-...  "not warranted"
```

Now approve it from a **completely separate process** — this is the real test, the same "genuinely a new `uv run`" bar [`stage06-sessions-langgraph`](../stage06-sessions-langgraph/) and [`stage09`'s own SqliteSaver choice] both use to prove real persistence, not just a variable staying in scope:

```bash
uv run agent.py --approve 6e7a633a-...
```

```
  [execute_high_risk_action] sending refund of $45.00

Refund of $45.00 sent. Reason: The item was received in a damaged condition.
```

Start a second request and reject it instead — the side effect never runs:

```bash
uv run agent.py "refund the customer $999 for a suspicious claim"
uv run agent.py --reject <thread_id> "needs manual investigation first"
```

List every request and its current status:

```bash
uv run agent.py --list
```

```
3 request(s):
  6e7a633a-...  [approved+executed]  $45.00 -- 'The item was received in a damaged condition.'
  b4294e14-...  [rejected]           $999.00 -- "...suspicious..."
  c6c4e73b-...  [pending approval]   $25.00 -- '...'
```

## What to look at closely

- **`resume()` calls `app.update_state(config, {...})` then `app.invoke(None, config=config)`** — passing `None` as the input is what tells LangGraph "don't start over, continue from the checkpoint." This is the exact two-call sequence from Stage 9's brief, not a simplification of it.
- **`resume()` checks `existing.values.get("executed")` before doing anything** — approving (or rejecting) an already-resolved thread_id is a no-op, not a re-execution. A real approval system needs this: a webhook firing twice should never send a refund twice.
- **`list_requests()` derives status (`pending` / `approved+executed` / `rejected`) entirely from state fields already written by the graph** — there's no separate status table to keep in sync; the checkpoint *is* the source of truth, same principle as [`stage06`'s `list_sessions()`](../stage06-sessions-langgraph/README.md).
- **The interrupt is declared once, at compile time** (`interrupt_before=[...]`), not as an `if` check inside `execute_high_risk_action` — the node itself has no idea it's gated; the graph's compilation is what enforces the pause. This keeps the high-risk node's code identical to what it would be without any approval step at all.

## Where this goes next

This is Stage 8's **human-checkpoint** node type — [`stage08-supervisor-langgraph`](../stage08-supervisor-langgraph/)'s node-type table lists it, this example builds it. A real system would combine both: a supervisor graph with an interrupt before any worker's high-risk action, not just a single linear draft → execute chain.
