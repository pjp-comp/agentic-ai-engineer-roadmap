[← Back to roadmap](../../README.md)

# Stage 1 — Async Fan-Out, Timeouts, and a Circuit Breaker

[Stage 1](../../docs/stage-01-python-async.md) is the one stage in this roadmap with no LLM in it at all — it's the async plumbing every agent in this repo sits on top of. This example builds that stage's brief literally: a FastAPI endpoint that calls three mock "tool" coroutines concurrently, with a per-call timeout, partial-failure handling, and a circuit breaker extension.

**No LLM, no Ollama, no Claude.** Every other example in this repo assumes the reflexes this one teaches are already automatic.

## How it works

```python
@app.post("/query")
async def query(req: QueryRequest):
    results = await asyncio.gather(
        call_tool("search", search(req.q)),
        call_tool("weather", weather(req.loc)),
        call_tool("calc", calc(req.expr)),
    )
    return dict(results)
```

Three mock tools, each simulating a different real failure mode:

- **`search`** — usually fast, occasionally slow enough to blow past the 1.5s timeout.
- **`weather`** — ~30% of calls raise a `ToolError` outright (a simulated downstream 500).
- **`calc`** — always fast and reliable. The control case: even when `search` or `weather` fails, `calc`'s real result still comes back — one slow/failing tool never drags the whole request down.

`call_tool()` wraps every call in `asyncio.wait_for(..., timeout=1.5)` and catches both `asyncio.TimeoutError` and `ToolError`, returning `{"error": ...}` for that one tool instead of letting the exception propagate and fail the whole endpoint.

**The circuit breaker (the brief's stated extension):** after 3 consecutive failures, a tool's circuit "opens" and further calls are skipped outright for 30 seconds — returning a fast `{"error": "circuit open..."}` instead of waiting out another timeout. This is the standard fix for "a flaky downstream dependency shouldn't make every request pay its timeout cost."

## Run it

```bash
cd examples/stage01-async-fanout
uv run uvicorn app:app --reload
```

In another terminal:

```bash
curl -X POST localhost:8000/query \
    -H "Content-Type: application/json" \
    -d '{"q": "async python", "loc": "Paris", "expr": "2+2"}'
```

Expected — `calc` always succeeds; `search`/`weather` vary run to run:

```json
{"search":{"tool":"search","query":"async python","results":["result for 'async python'"],"took":0.67},
 "weather":{"tool":"weather","location":"Paris","forecast":"22C, clear"},
 "calc":{"tool":"calc","expression":"2+2","result":4}}
```

Call it 4-5 times in a row and watch `search`'s circuit open once it fails 3 times consecutively:

```json
{"search":{"error":"circuit open -- tool skipped after repeated failures"}, ...}
```

## What to look at closely

- **`asyncio.TimeoutError`'s `str()` is empty** — a real gotcha this example ran into: catching it and returning `str(e)` as the error message produces `{"error": ""}`, which tells the caller nothing. The fix is to build a real message (`f"{name} timed out after {TIMEOUT_SECONDS}s"`) rather than trust the exception's string form.
- **The circuit breaker is per-tool, not global** — `search` opening its circuit has no effect on `weather` or `calc`. A systemic outage (all three failing) would open all three circuits independently, which is the correct behavior: each dependency's health is tracked separately.
- **`calc` uses a bare `eval()`** — deliberately, because this is a local demo with no untrusted input reaching it. [`stage12-guardrails`](../stage12-guardrails/) builds the sandboxed version of the same idea for when a code-execution tool's input might be adversarial — read that one before reusing `eval()` anywhere real.
- **`asyncio.gather()` runs all three tools concurrently, not sequentially** — the whole endpoint's latency is bounded by the *slowest* tool call, not the sum of all three. This is the actual point of the stage: a blocking, sequential version of this endpoint would take 3x as long even when nothing fails.

## Where this goes next

Every `agent.py` in `stage02` onward assumes calls to a model or a tool can hang, fail, or need a timeout — this stage's `call_tool()` pattern is the ancestor of the retry/timeout logic those examples build on top. [Stage 13](../../docs/stage-13-deployment.md#idempotent-tools--tool-caching--why-retries-are-dangerous-by-default) picks the same "what happens on a retry" question back up for tools with real side effects — see [`stage13-idempotent-tools`](../stage13-idempotent-tools/).
