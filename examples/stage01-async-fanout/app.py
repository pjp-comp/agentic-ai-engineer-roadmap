"""
Async fan-out with per-call timeout + circuit breaker (Stage 1 --
docs/stage-01-python-async.md), matching that stage's brief exactly:

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

No LLM, no agent, on purpose -- this stage is entirely about async
plumbing (concurrency, timeouts, partial-failure handling) BEFORE any
model calls get layered on top. Every example from stage02 onward in this
repo assumes this stage's reflexes are already automatic.

Three mock "tool" coroutines, each simulating a different real-world
failure mode:
  - search()  -- usually fast, occasionally slow enough to hit the timeout
  - weather() -- occasionally raises a ToolError outright (a "the
                 downstream API returned 500" simulation)
  - calc()    -- always fast and reliable -- the control case, so a
                 request with a slow/failing tool still returns THIS
                 tool's real result, not get dragged down by the others

Extension beyond the base brief: a circuit breaker per tool. After 3
consecutive failures (timeout or ToolError), that tool is skipped
outright for 30s and returns a fast "circuit open" response instead of
waiting out another timeout -- the standard fix for "a flaky downstream
dependency shouldn't make every request pay its timeout cost."

Usage:
    uv run uvicorn app:app --reload
    curl -X POST localhost:8000/query \
        -H "Content-Type: application/json" \
        -d '{"q": "async python", "loc": "Paris", "expr": "2+2"}'
"""

import asyncio
import random
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel

TIMEOUT_SECONDS = 1.5
FAILURE_THRESHOLD = 3       # consecutive failures before a tool's circuit opens
CIRCUIT_RESET_SECONDS = 30  # how long a tool stays skipped once its circuit opens


class ToolError(Exception):
    pass


class QueryRequest(BaseModel):
    q: str
    loc: str
    expr: str


class CircuitBreaker:
    """One breaker per tool name. Tracks consecutive failures; once the
    threshold is hit, the circuit "opens" and calls are short-circuited
    (skipped, not even attempted) until CIRCUIT_RESET_SECONDS has passed.
    """

    def __init__(self):
        self._consecutive_failures: dict[str, int] = {}
        self._opened_at: dict[str, float] = {}

    def is_open(self, name: str) -> bool:
        opened_at = self._opened_at.get(name)
        if opened_at is None:
            return False
        if time.monotonic() - opened_at >= CIRCUIT_RESET_SECONDS:
            # Reset window elapsed -- give the tool another chance instead
            # of skipping it forever.
            del self._opened_at[name]
            self._consecutive_failures[name] = 0
            return False
        return True

    def record_success(self, name: str) -> None:
        self._consecutive_failures[name] = 0

    def record_failure(self, name: str) -> None:
        count = self._consecutive_failures.get(name, 0) + 1
        self._consecutive_failures[name] = count
        if count >= FAILURE_THRESHOLD:
            self._opened_at[name] = time.monotonic()


breaker = CircuitBreaker()


# --- Mock tool coroutines -- deliberately unreliable, on purpose --------

async def search(q: str) -> dict:
    delay = random.uniform(0.2, 2.5)   # sometimes exceeds TIMEOUT_SECONDS
    await asyncio.sleep(delay)
    return {"tool": "search", "query": q, "results": [f"result for {q!r}"], "took": round(delay, 2)}


async def weather(loc: str) -> dict:
    await asyncio.sleep(0.3)
    if random.random() < 0.3:          # ~30% simulated downstream 500
        raise ToolError(f"weather service unavailable for {loc!r}")
    return {"tool": "weather", "location": loc, "forecast": "22C, clear"}


async def calc(expr: str) -> dict:
    await asyncio.sleep(0.1)           # always fast -- the reliable control case
    try:
        # eval() is fine here: this is a local demo tool, not a
        # production expression evaluator taking untrusted input over the
        # network without sandboxing (see stage12-guardrails for that).
        return {"tool": "calc", "expression": expr, "result": eval(expr, {"__builtins__": {}})}
    except Exception as e:
        raise ToolError(f"invalid expression {expr!r}: {e}")


TOOLS = {"search": search, "weather": weather, "calc": calc}


async def call_tool(name: str, coro) -> tuple[str, dict]:
    if breaker.is_open(name):
        print(f"  [circuit] {name} is OPEN -- skipping call")
        return name, {"error": "circuit open -- tool skipped after repeated failures"}
    try:
        result = await asyncio.wait_for(coro, timeout=TIMEOUT_SECONDS)
        breaker.record_success(name)
        return name, result
    except asyncio.TimeoutError:
        # asyncio.TimeoutError's str() is empty -- build a message that
        # actually says what happened instead of an empty error field.
        breaker.record_failure(name)
        message = f"{name} timed out after {TIMEOUT_SECONDS}s"
        print(f"  [failure] {message}")
        return name, {"error": message}
    except ToolError as e:
        breaker.record_failure(name)
        print(f"  [failure] {name}: {e}")
        return name, {"error": str(e)}


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(lifespan=lifespan)


@app.post("/query")
async def query(req: QueryRequest) -> dict:
    results = await asyncio.gather(
        call_tool("search", search(req.q)),
        call_tool("weather", weather(req.loc)),
        call_tool("calc", calc(req.expr)),
    )
    return dict(results)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}
