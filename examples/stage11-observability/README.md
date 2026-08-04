[← Back to roadmap](../../README.md)

# Stage 11 — Tracing the Stage 8 Supervisor Graph, For Real

[Stage 11](../../docs/stage-11-observability.md)'s brief is "instrument the Stage 8 supervisor graph with tracing so every node emits a span with input/output, token counts, latency, and cost." This example does exactly that — it's [`stage08-supervisor-langgraph`](../stage08-supervisor-langgraph/)'s graph, unchanged in shape, with every node wrapped in a real span. No LangSmith account or API key required: spans are written to a local JSONL file, which is enough to build the same dashboard concepts (p95 latency, $/run, span-by-span lookup) a hosted tracing backend would show you.

**Ollama only** — `llama3.2:3b`, no Claude path.

## How it works

`tracing.py` defines a `Span` dataclass mirroring what a real OpenTelemetry/LangSmith span carries: `trace_id` (one per `graph.invoke()` call), `span_id` (one per node), timing, input/output (truncated for readability), and token counts. `Tracer.span()` is a context manager — wrap any node's body in `with tracer.span("name", input_data=...) as span:` and it's automatically timed, and written to `.traces.jsonl` on exit (even on error).

`agent.py` is `stage08`'s graph with one line added per node — a `with tracer.span(...)` wrapping the existing logic. Token counts come from `ChatOllama`'s real `response_metadata` (`prompt_eval_count`, `eval_count`), not placeholders.

`dashboard.py` reads `.traces.jsonl` and gives two views:

- **Summary** — p95 latency and total tokens/cost per node (across every trace), plus total latency/cost per trace. This is Stage 11's "one dashboard view."
- **`--trace <id>`** — every span in one run, in order, with full input/output. This is Stage 11's "Done when": given a bad output, find the exact span that caused it.

## Run it

```bash
cd examples/stage11-observability
ollama pull llama3.2:3b   # one-time, ~2GB, shared with other examples
uv run agent.py "the tradeoffs of microservices vs a monolith"
```

Prints the usual graph output, plus a trace_id:

```
  [supervisor] routing -> researcher
  [researcher] researching '...' (attempt 1)
  [critic] approved -- handoff to writer
  ...

(trace_id: a66e8c13-6ca4-40cb-aa38-536e1c9ca3e8)
Inspect it: uv run dashboard.py --trace a66e8c13-6ca4-40cb-aa38-536e1c9ca3e8
```

Run it a couple more times with different topics, then look at the dashboard:

```bash
uv run dashboard.py
```

```
2 trace(s), 12 span(s) total

node          calls   p95 latency (ms)   total tokens    total $
critic            2                0.1              0     0.0000
researcher        2             6303.3            504     0.0000
supervisor        6                0.1              0     0.0000
writer            2             2721.9            612     0.0000

trace_id                                nodes   total latency (ms)    total $
a66e8c13-...                                6               8267.0     0.0000
4575e0fa-...                                6               7374.8     0.0000
```

Then drill into one specific run:

```bash
uv run dashboard.py --trace a66e8c13-6ca4-40cb-aa38-536e1c9ca3e8
```

Shows every node's real input and output, in order — enough to spot exactly which node produced a bad result.

## What to look at closely

- **`supervisor` and `critic` show `0.1ms` latency and `0` tokens** — correctly reflects that they're router/guardrail nodes (Stage 8's node-type table), not LLM calls. A dashboard that only tracked "time per node" without this distinction would make it look like the graph has 4 expensive steps instead of 2 cheap ones and 2 expensive ones — the summary view is what makes that visible.
- **Cost is always `$0.0000`** — Ollama is free and local. `COST_PER_1K_TOKENS` is a real constant in `tracing.py`, kept at `0.0` here on purpose: if `LOCAL_MODEL` were ever swapped for a paid API (Stage 13's deployment concern), the dashboard code wouldn't need to change, only that one constant.
- **Spans are written even on error** — `Tracer.span()`'s `finally` block writes the span whether the node succeeded or raised, with `span.error` set in the failure case. A crash mid-run still leaves a usable partial trace, not a silent gap.
- **This is a flat trace, not a nested one** — every span's `parent_id` is the trace root, because this graph's nodes don't call sub-graphs. A graph with nested sub-agents (Stage 8's more complex topologies) would need real span nesting, which this minimal tracer doesn't build — worth knowing as the honest limit of "the smallest real version," not something papered over.

## Where this goes next

A real deployment would export these same spans to LangSmith, Langfuse, or an OpenTelemetry collector instead of a local JSONL file — the *shape* of the data (trace_id, span_id, input/output, tokens, latency) is what those backends expect, so swapping `Tracer._write()`'s destination is the actual migration path, not a rewrite. See [Stage 13](../../docs/stage-13-deployment.md) for what changes once this graph is actually running behind a deployed service.
