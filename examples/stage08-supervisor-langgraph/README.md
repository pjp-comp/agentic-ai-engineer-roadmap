[← Back to roadmap](../../README.md)

# Stage 8 — A Supervisor, Two Workers, and a Guardrail at the Handoff

[Stage 8](../../docs/stage-08-multi-agent.md)'s brief is a supervisor node routing between a researcher and a writer worker, expressed as a LangGraph `StateGraph`. This example builds exactly that graph, plus one addition the doc calls out as a distinct, worthwhile pattern: a **critic node sitting between researcher and writer**, rejecting a bad handoff instead of letting it propagate.

**Ollama only** — `llama3.2:3b`, no Claude path.

## How it works

```
START -> supervisor -> (route_next) -> researcher -> critic -> supervisor
                            |-> writer -> supervisor
                            |-> done -> END
```

- **`supervisor`** — a **router node** (Stage 8's node-type table): plain Python, no LLM call. `route_next()` decides who goes next purely from what's already in state. This is the cost-saving point Stage 8 makes explicitly — a router doesn't need to reason, so it shouldn't be another expensive LLM call.
- **`researcher`** — an **agent node**: one LLM call, writes `research_notes` into shared state.
- **`critic`** — a **guardrail node**, sitting on the edge between `researcher` and the supervisor. No LLM call — it's a cheap length check. Rejects research that's empty or too short and routes back for another attempt, instead of letting the writer draft from garbage. This is Stage 8's ["Guardrails at the handoff, not just at the end"](../../docs/stage-08-multi-agent.md#guardrails-at-the-handoff-not-just-at-the-end) section, built as an actual node instead of just described.
- **`writer`** — an **agent node**: drafts from `research_notes`. It never calls `researcher` directly — its only inputs are graph state and its own code, the same structural point [`stage06-sessions-langgraph`](../stage06-sessions-langgraph/) makes about inter-agent communication.

`MAX_RESEARCH_ATTEMPTS` caps how many times the critic can send work back — the same graceful-degradation idea as [`stage05-rag-langgraph`](../stage05-rag-langgraph/)'s `MAX_RETRIES`: forward whatever exists rather than loop forever on research that keeps failing the check.

## Run it

```bash
cd examples/stage08-supervisor-langgraph
ollama pull llama3.2:3b   # one-time, ~2GB, shared with other examples
uv run agent.py "the tradeoffs of microservices vs a monolith"
```

Expected — supervisor routes to the researcher, the critic approves on the first pass, the supervisor routes to the writer, then ends:

```
  [supervisor] routing -> researcher
  [researcher] researching 'the tradeoffs of microservices vs a monolith' (attempt 1)
  [critic] approved -- handoff to writer
  [supervisor] routing -> writer
  [writer] drafting from approved research notes
  [supervisor] routing -> done

Here is a 2-sentence summary:
...
```

## What to look at closely

- **`route_next()` is the entire supervisor decision**, expressed as a plain function returning a string that `add_conditional_edges` uses — not an `if/else` buried inside a bigger function. Same "control flow as data" point [`stage05-rag-langgraph`](../stage05-rag-langgraph/)'s `route_after_grading()` makes.
- **The critic sits *on the edge*, not inside either agent** — `researcher_node` doesn't know or care that its output gets checked; `writer_node` never sees rejected research at all. Putting validation at the boundary between two nodes, rather than folding it into one of them, is what makes it reusable if a third worker ever needs the same check.
- **`supervisor_node` records its own decision into state (`next_step`)** even though `add_conditional_edges` calls `route_next()` separately to actually route — this is a deliberate observability seam: Stage 11's tracing needs to know *which* node the supervisor picked and when, not just infer it from which node ran next.
- **Two different failure responses to the same problem** — a `researcher` that repeatedly produces thin output eventually gets forwarded anyway (`MAX_RESEARCH_ATTEMPTS`), not rejected forever. An agent that can reject work indefinitely is a liability, the same lesson as Stage 7's `MAX_ITERS`.

## Where this goes next

See [Stage 8's graph-engineering section](../../docs/stage-08-multi-agent.md#graph-engineering-in-depth--why-the-topology-is-a-first-class-design-decision) for what changes once a graph outgrows this shape — joins for parallel branches, human checkpoints, more heterogeneous node types. [`stage09-human-in-the-loop`](../stage09-human-in-the-loop/) builds exactly the human-checkpoint node type this stage's node-type table lists but doesn't build.
