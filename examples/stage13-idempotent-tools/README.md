[← Back to roadmap](../../README.md)

# Stage 13 — Idempotency Keys and Tool Caching, Proven With a Simulated Crash

[Stage 13](../../docs/stage-13-deployment.md)'s "Idempotent tools + tool caching" section is the one part of that stage that's actually a runnable pattern rather than infrastructure (Kubernetes, canary deploys) that doesn't fit a local example. This builds both techniques from that section, plus the proof the doc's own "Done when" bar asks for: killing a call right after a side effect runs, then retrying, and confirming the action didn't happen twice.

**No LLM required** — this is entirely about the tool layer underneath an agent, not the model. It runs with plain Python, no Ollama, no Claude — the same way [`stage01-async-fanout`](../stage01-async-fanout/) is about the layer underneath agents rather than an agent itself.

## How it works

- **`idempotency_key(tool_name, tool_input)`** — a deterministic SHA-256 hash of the tool name + sorted input. Same call, same key, every time — a genuine retry of the *same logical action* collides with the original on purpose. (A fresh `uuid4()` per call would do the opposite: make every retry look like a brand-new action, which is exactly the bug this technique prevents.)
- **`ChargeLedger.charge_card()`** — Stage 13's own example tool. Checks `seen_keys` for the idempotency key before calling `_actually_charge()` (the real, external side effect). A second call with identical `(amount, customer_id)` returns the cached result instead of charging again.
- **`CachedTool`** — a TTL-based wrapper for read-only tools. Distinct purpose from idempotency: `get_weather("Paris")` run twice is harmless either way, so this exists purely to save a redundant network round-trip within the TTL window, not to prevent a duplicate side effect.

## Run it

```bash
cd examples/stage13-idempotent-tools
uv run agent.py
```

The crash simulation:

```
--- First attempt ---
  [ledger] ACTUALLY CHARGING $49.99 to cust_88 (charge #1)
  [simulated crash] process dies here -- caller never saw the response, doesn't know if it succeeded

--- Retry (same logical call, because the caller doesn't know the first one worked) ---
  [ledger] idempotency key 40c331f1... already seen -- returning cached result, NOT charging again
  [retry result] {'status': 'charged', 'amount': 49.99, 'customer_id': 'cust_88'}

Verified: 1 real charge(s) despite 2 calls -- idempotency key worked.
```

The caching demo:

```
--- Tool-result caching (read-only tool, no side effect to protect) ---
Paris: 18C, partly cloudy
  [cache] hit for _get_weather({'city': 'Paris'}) -- skipping real call
Paris: 18C, partly cloudy
Tokyo: 18C, partly cloudy

get_weather() actually ran 2 time(s) for 3 calls (2 unique cities).
```

## What to look at closely

- **`ChargeLedger.seen_keys` is what prevents the double charge, not agent discipline** — `simulate_crash_then_retry()` calls `charge_card()` twice with identical arguments, deliberately not trying to be clever about detecting its own crash. The ledger (standing in for a real payments backend's own idempotency-key tracking, like Stripe's `Idempotency-Key` header) is what catches the retry — the calling code is allowed to be dumb and retry blindly, which is the actual point: production retries are routine, not something you can rely on an agent to reason its way around.
- **`idempotency_key()` uses `json.dumps(..., sort_keys=True)`** — sorting keys before hashing matters: `{"amount": 5, "customer_id": "x"}` and `{"customer_id": "x", "amount": 5}` must produce the *same* key even though Python dict insertion order differs between them, or a harmless serialization difference would look like a different logical action.
- **`CachedTool` and `ChargeLedger` solve different problems with a similar-looking key function** — worth noticing that `idempotency_key()` is reused for both, but the two structures around it do opposite things: the ledger's cache is there to be *hit* on purpose (prevent a repeat), the tool cache's TTL exists so it *expires* on purpose (a weather report from 10 minutes ago shouldn't be trusted forever).
- **Not every tool needs an idempotency key** — `calculate` (from [`stage01-async-fanout`](../stage01-async-fanout/) or [`stage12-guardrails`](../stage12-guardrails/)) has no side effect, so running it twice is harmless; only `CachedTool`'s latency-saving applies, not this file's `ChargeLedger` pattern. Reserve idempotency keys specifically for tools that write, charge, send, or delete.

## Where this goes next

This is the last piece of local, runnable logic from [Stage 13](../../docs/stage-13-deployment.md) — the rest of that stage (containerizing the [Stage 9](../../docs/stage-09-human-in-the-loop.md) agent service, a CI gate running [Stage 10](../../docs/stage-10-evaluation-qa.md)'s eval suite, canary deploys with automatic rollback) is infrastructure that doesn't have a meaningful "run it locally" form — see that stage's doc directly for the CI/Kubernetes brief.
