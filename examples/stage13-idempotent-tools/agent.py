"""No-model demonstration of retry identity and read caching.
The ledger stays in memory: a repeated call models a lost response, not an
actual process crash. Production needs durable, atomic backend idempotency.
"""

import hashlib
import json
import time
from dataclasses import dataclass, field


def idempotency_key(tool_name: str, tool_input: dict) -> str:
    """Content fingerprint for read caching and payload consistency checks."""
    payload = json.dumps({"tool": tool_name, "input": tool_input}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


# --- 1. Idempotency keys — side-effecting tool -----------------------------

@dataclass
class ChargeLedger:
    """Single-process mock backend, not crash-safe or concurrency-safe."""
    seen_keys: dict[str, dict] = field(default_factory=dict)
    actual_charges: list[dict] = field(default_factory=list)   # the real, external side effect log

    def charge_card(self, amount: float, customer_id: str, action_id: str) -> dict:
        if not action_id:
            raise ValueError("action_id is required")
        fingerprint = idempotency_key("charge_card", {"amount": amount, "customer_id": customer_id})
        if action_id in self.seen_keys:
            previous = self.seen_keys[action_id]
            if previous["fingerprint"] != fingerprint:
                raise ValueError("action_id reused with different inputs")
            print(f"  [ledger] action {action_id} already seen -- returning previous result")
            return previous["result"]
        result = self._actually_charge(amount, customer_id)
        self.seen_keys[action_id] = {"fingerprint": fingerprint, "result": result}
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
            if time.monotonic() - cached_at < self.ttl:
                print(f"  [cache] hit for {self.fn.__name__}({kwargs}) -- skipping real call")
                return result
        self.call_count += 1
        result = self.fn(**kwargs)
        self._cache[key] = (time.monotonic(), result)
        return result


def _get_weather(city: str) -> str:
    # Simulates a real network round-trip a cache would save you from repeating.
    time.sleep(0.3)
    return f"{city}: 18C, partly cloudy"


get_weather = CachedTool(_get_weather, ttl_seconds=5.0)


# --- 3. The lost-response retry demonstration -----------------------------------------

def simulate_lost_response_then_retry(ledger: ChargeLedger) -> None:
    """Simulate a lost response while the mock backend remains alive."""
    amount, customer_id = 49.99, "cust_88"

    print("\n--- First attempt ---")
    ledger.charge_card(amount, customer_id, action_id="purchase-001")
    print("  [lost response] caller did not receive the result; mock backend remains in memory")

    print("\n--- Retry (same logical call, because the caller doesn't know the first one worked) ---")
    result = ledger.charge_card(amount, customer_id, action_id="purchase-001")
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
    simulate_lost_response_then_retry(ledger)
    ledger.charge_card(49.99, "cust_88", action_id="purchase-002")
    assert len(ledger.actual_charges) == 2, "A distinct purchase must not be suppressed"
    demonstrate_caching()
