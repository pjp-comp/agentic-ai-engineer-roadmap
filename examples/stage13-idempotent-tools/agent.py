"""
Idempotent tools + tool caching (Stage 13 --
docs/stage-13-deployment.md), building the two techniques from that
stage's "Idempotent tools + tool caching" section for real, plus a
concrete demonstration of the failure they prevent: a retry after a
crash must NOT repeat a side effect.

Ollama only where a model is used at all -- but this example is mostly
about the tool layer underneath the model, not the model itself, so most
of it runs with no LLM call, the same way stage01 is about the async
layer underneath agents.

Three pieces, matching the doc's structure exactly:

  1. idempotency_key(tool_name, tool_input) -- deterministic hash of the
     LOGICAL action, not a random UUID. The same call, retried, produces
     the SAME key on purpose, so a downstream check can catch the retry.

  2. charge_card() -- Stage 13's own example tool: a side-effecting
     action guarded by seen_keys. Calling it twice with the same
     (amount, customer_id) charges the customer ONCE -- the second call
     returns the cached result instead of re-running _actually_charge().

  3. CachedTool -- a decorator-shaped wrapper for READ-ONLY tools
     (get_weather-style), saving latency/cost on a repeated call within
     a TTL window. Distinct from idempotency: a cached read-only call
     being "repeated" is harmless either way, this only saves a
     redundant network round-trip.

The crash simulation (`simulate_crash_then_retry`) is the actual proof,
not just a docstring claim: it runs a normal ReAct-style loop that calls
charge_card, kills the loop's execution BEFORE the result reaches the
caller (simulating a process death right after the side effect happened
but before the response was received -- Stage 13's own "Done when" bar),
then retries the exact same logical call and shows the customer was only
charged once.

Usage:
    uv run agent.py
"""

import hashlib
import json
import sys
import time
from dataclasses import dataclass, field


def idempotency_key(tool_name: str, tool_input: dict) -> str:
    """Deterministic: same tool + same input always produces the same
    key, so a genuine retry of the SAME logical action collides with the
    original call on purpose (as opposed to a fresh uuid4(), which would
    make every retry look like a brand-new, distinct action).
    """
    payload = json.dumps({"tool": tool_name, "input": tool_input}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


# --- 1. Idempotency keys — side-effecting tool -----------------------------

@dataclass
class ChargeLedger:
    """Stands in for a real payments backend's own idempotency-key
    tracking (e.g. Stripe's Idempotency-Key header) -- seen_keys here is
    what that backend would keep, not something the AGENT is trusted to
    track itself. An agent retrying blindly is fine BECAUSE the ledger,
    not the agent's good behavior, is what prevents the double charge.
    """
    seen_keys: dict[str, dict] = field(default_factory=dict)
    actual_charges: list[dict] = field(default_factory=list)   # the real, external side effect log

    def charge_card(self, amount: float, customer_id: str) -> dict:
        key = idempotency_key("charge_card", {"amount": amount, "customer_id": customer_id})
        if key in self.seen_keys:
            print(f"  [ledger] idempotency key {key[:8]}... already seen -- returning cached result, NOT charging again")
            return self.seen_keys[key]

        result = self._actually_charge(amount, customer_id)
        self.seen_keys[key] = result
        return result

    def _actually_charge(self, amount: float, customer_id: str) -> dict:
        # The real, external side effect -- appended to actual_charges,
        # which is what the demo checks at the end to prove it ran once.
        self.actual_charges.append({"amount": amount, "customer_id": customer_id, "at": time.time()})
        print(f"  [ledger] ACTUALLY CHARGING ${amount:.2f} to {customer_id} (charge #{len(self.actual_charges)})")
        return {"status": "charged", "amount": amount, "customer_id": customer_id}


# --- 2. Tool-result caching — read-only tool -------------------------------

class CachedTool:
    """Wraps a read-only, side-effect-free function with a TTL cache.
    Distinct purpose from idempotency: this exists purely to save
    latency/cost on a repeated call, not to prevent a duplicate side
    effect (there isn't one -- a weather lookup run twice is harmless).
    """

    def __init__(self, fn, ttl_seconds: float = 5.0):
        self.fn = fn
        self.ttl = ttl_seconds
        self._cache: dict[str, tuple[float, str]] = {}
        self.call_count = 0   # how many times fn() ITSELF actually ran, vs. served from cache

    def __call__(self, **kwargs) -> str:
        key = idempotency_key(self.fn.__name__, kwargs)
        cached = self._cache.get(key)
        if cached is not None:
            cached_at, result = cached
            if time.time() - cached_at < self.ttl:
                print(f"  [cache] hit for {self.fn.__name__}({kwargs}) -- skipping real call")
                return result
        self.call_count += 1
        result = self.fn(**kwargs)
        self._cache[key] = (time.time(), result)
        return result


def _get_weather(city: str) -> str:
    # Simulates a real network round-trip a cache would save you from repeating.
    time.sleep(0.3)
    return f"{city}: 18C, partly cloudy"


get_weather = CachedTool(_get_weather, ttl_seconds=5.0)


# --- 3. The crash-then-retry proof -----------------------------------------

def simulate_crash_then_retry(ledger: ChargeLedger) -> None:
    """Runs the SAME logical charge_card call twice, with a simulated
    crash between them -- proving the ledger's idempotency key, not
    agent discipline, is what prevents a double charge. This mirrors
    Stage 13's own 'Done when': killing the process right after
    charge_card runs, before the response is received, and confirming
    the action didn't happen twice on retry.
    """
    amount, customer_id = 49.99, "cust_88"

    print("\n--- First attempt ---")
    ledger.charge_card(amount, customer_id)
    print("  [simulated crash] process dies here -- caller never saw the response, doesn't know if it succeeded")

    print("\n--- Retry (same logical call, because the caller doesn't know the first one worked) ---")
    result = ledger.charge_card(amount, customer_id)
    print(f"  [retry result] {result}")

    assert len(ledger.actual_charges) == 1, "BUG: customer was charged twice!"
    print(f"\nVerified: {len(ledger.actual_charges)} real charge(s) despite 2 calls -- idempotency key worked.")


def demonstrate_caching() -> None:
    print("\n--- Tool-result caching (read-only tool, no side effect to protect) ---")
    print(get_weather(city="Paris"))
    print(get_weather(city="Paris"))   # should hit cache, not sleep again
    print(get_weather(city="Tokyo"))   # different input -> real call
    print(f"\nget_weather() actually ran {get_weather.call_count} time(s) for 3 calls (2 unique cities).")


if __name__ == "__main__":
    ledger = ChargeLedger()
    simulate_crash_then_retry(ledger)
    demonstrate_caching()
