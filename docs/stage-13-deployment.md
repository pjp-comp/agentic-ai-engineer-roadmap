[← Stage 12](stage-12-security-guardrails.md) · Stage 13 of 14 · **Next:** [Stage 14 →](stage-14-portfolio.md)

# Stage 13 — Production Deployment

vLLM/SGLang · Kubernetes scaling · CI/CD for agents · canary releases · rollback strategies · idempotent tools + tool caching

## Why this matters

Kubernetes earns its keep only once you need horizontal scaling, multi-tenant isolation, or rolling deploys — not by default. If self-hosting open-weight models, vLLM (broad compatibility, Apache-2.0) and SGLang (higher throughput, structured-generation-first) are the two serving engines worth knowing. Version model weights, serving config, and prompt templates independently — all three affect output quality and drift separately.

## Brief

Containerize the Stage 9 agent service; write a CI pipeline that runs the Stage 10 eval suite as a merge gate; deploy behind a canary (5% traffic) with automatic rollback if error rate or eval pass-rate regresses.

```yaml
# CI gate (excerpt)
- run: pytest tests/eval_suite.py --min-pass-rate=0.92
- run: docker build -t agent:$SHA .
- run: kubectl set image deployment/agent-canary agent=agent:$SHA
- run: ./scripts/watch_canary.sh --rollback-on-error-rate=0.02
```

## Idempotent tools + tool caching — why retries are dangerous by default

Once an agent is live, retries are unavoidable: a network blip, a rate limit, a crashed worker mid-loop — all of these mean "run the same tool call again" is a routine event, not an edge case. The problem is that most tools are **not safe to run twice**. An agent that retries a `charge_card` or `send_email` tool call after a timeout — not knowing whether the first call actually succeeded before the timeout — can double-charge a customer or send a duplicate email. This is a production incident category of its own, distinct from the model-quality concerns the rest of this roadmap focuses on.

Two techniques fix this, and you generally want both:

**1. Idempotency keys.** Every tool call that has a side effect should carry a deterministic identifier derived from its inputs (not a random UUID generated fresh each call — it must be the *same* key on a retry of the *same* logical action). The downstream system checks the key before acting: if it's seen this key before, it returns the previous result instead of repeating the side effect.

```python
import hashlib
import json

def idempotency_key(tool_name: str, tool_input: dict) -> str:
    # Deterministic: same tool + same input always produces the same key,
    # so a genuine retry collides with the original call on purpose.
    payload = json.dumps({"tool": tool_name, "input": tool_input}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()

def charge_card(amount: float, customer_id: str, seen_keys: dict) -> str:
    key = idempotency_key("charge_card", {"amount": amount, "customer_id": customer_id})
    if key in seen_keys:
        return seen_keys[key]  # already ran — return the prior result, don't charge again
    result = _actually_charge(amount, customer_id)
    seen_keys[key] = result
    return result
```

**2. Tool-result caching.** Distinct from idempotency (which prevents *duplicate side effects*), caching also saves latency and API cost on read-only or expensive tools — a repeated `get_weather("Paris")` or a repeated database lookup within the same conversation doesn't need to hit the network twice.

```python
class CachedTool:
    def __init__(self, fn, ttl_seconds=300):
        self.fn = fn
        self.ttl = ttl_seconds
        self._cache: dict[str, tuple[float, str]] = {}

    def __call__(self, **kwargs) -> str:
        key = idempotency_key(self.fn.__name__, kwargs)
        if key in self._cache:
            cached_at, result = self._cache[key]
            if time.time() - cached_at < self.ttl:
                return result
        result = self.fn(**kwargs)
        self._cache[key] = (time.time(), result)
        return result
```

**Not every tool needs this.** Read-only, side-effect-free tools (a calculator, a search) don't need idempotency keys — running them twice is harmless, though caching them still saves cost. Reserve idempotency keys specifically for tools with real side effects: anything that writes, charges, sends, or deletes.

**Runnable version:** [`examples/stage13-idempotent-tools/`](../examples/stage13-idempotent-tools/) builds both `idempotency_key()` and `CachedTool` for real, then simulates the exact failure this section describes — a process dying right after a `charge_card` call, before the caller sees the response — and proves the retry that follows doesn't double-charge. No LLM required; this is the tool layer underneath an agent, not the model.

## Sources

| Type | Resource |
|------|----------|
| Docs | [vLLM — Using Kubernetes (official docs)](https://docs.vllm.ai/en/stable/deployment/k8s/) |
| Guide | [vLLM Production Deployment — complete 2026 guide](https://www.sitepoint.com/vllm-production-deployment-guide-2026/) |
| Guide | [SGLang — the complete guide to high-performance LLM inference](https://inference.net/content/sglang-complete-guide/) |
| Reference | LiteLLM as an OpenAI-compatible gateway in front of multiple backends |
| Guide | [PADISO — Building idempotent tools for long-running agents](https://www.padiso.co/blog/building-idempotent-tools-for-long-running-agents/) |
| Guide | [MightyBot — Designing fault-tolerant AI agent pipelines: idempotency, retries, state management](https://mightybot.ai/blog/fault-tolerant-ai-agent-pipelines/) |

## Done when

A bad deploy auto-rolls-back before it reaches full traffic, with no manual intervention — and forcing a retry on a side-effecting tool call (e.g. killing the process right after `charge_card` runs, before the response is received) does not result in the action happening twice.

---
[← Stage 12 — Security + Guardrails](stage-12-security-guardrails.md) · [Back to roadmap](../README.md) · **Next:** [Stage 14 — Open Source + Portfolio →](stage-14-portfolio.md)
