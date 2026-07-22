[← Back to roadmap](../README.md) · Stage 01 of 13 · **Next:** [Stage 02 →](stage-02-llm-fundamentals.md)

# Stage 01 — Python + Async Foundations

`asyncio` · `FastAPI` · event-driven architecture · error handling · API integration patterns

## Why this first

Every agent is fundamentally a program juggling slow, unreliable I/O — model calls, tool calls, retries. If `async`/`await`, backpressure, and timeout handling aren't reflexive, everything built on top will be flaky in ways that are hard to diagnose later.

## Brief

Build a FastAPI service with one endpoint that fans out to three mock "tool" coroutines concurrently via `asyncio.gather`, applies a per-call timeout, and returns partial results if one tool fails instead of the whole request failing.

```python
@app.post("/query")
async def query(req: QueryRequest):
    async def call_tool(name, coro):
        try:
            return name, await asyncio.wait_for(coro, timeout=3.0)
        except (asyncio.TimeoutError, ToolError) as e:
            return name, {"error": str(e)}

    results = await asyncio.gather(
        call_tool("search", search(req.q)),
        call_tool("weather", weather(req.loc)),
        call_tool("calc", calculate(req.expr)),
    )
    return dict(results)
```

Extend it: add a circuit breaker so a tool that's failed 3× in a row gets skipped for 30s.

## Sources

| Type | Resource |
|------|----------|
| Docs | [FastAPI — Concurrency and async/await](https://fastapi.tiangolo.com/async/) |
| Docs | [Python — asyncio Tasks & coroutines](https://docs.python.org/3/library/asyncio-task.html) |
| Guide | [Mastering FastAPI: a complete learning roadmap](https://dev.to/prasanna_kumar/mastering-fastapi-a-complete-learning-roadmap-477o) |
| Tutorial | [FastAPI in 13 steps — Pydantic, async Postgres, JWT, Docker](https://tech-insider.org/fastapi-tutorial-python-rest-api-13-steps-2026/) |
| Tool | `tenacity` for retry/backoff, `httpx` for async HTTP clients |

## Done when

You can explain why a blocking call inside an `async def` stalls the whole event loop, and your endpoint above survives one tool hanging forever.

---
[← Back to roadmap](../README.md) · **Next:** [Stage 02 — LLM Fundamentals for Agents →](stage-02-llm-fundamentals.md)
