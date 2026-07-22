[← Stage 01](stage-01-python-async.md) · Stage 02 of 14 · **Next:** [Stage 03 →](stage-03-tool-calling.md)

# Stage 02 — LLM Fundamentals for Agents

Context management · model routing · token economics · latency tradeoffs · failure modes

## Why this matters

Agent cost and latency are dominated by model choice and context size, not your orchestration code. Knowing when a small/fast model beats a frontier one — and how prompt caching changes the economics — is what separates a demo from something affordable to run at volume.

## Brief

Build a router that classifies incoming requests by complexity (regex/heuristic first, cheap-model classifier second) and sends "simple" ones to a fast/cheap model, "complex" ones to a frontier model. Log tokens, latency, and $ per request for both paths.

```python
def route(task_complexity: str) -> str:
    return {
        "simple":  "claude-haiku-4-5",
        "complex": "claude-opus-4-8",
    }.get(task_complexity, "claude-sonnet-5")

# instrument every call:
# tokens_in, tokens_out, cache_read_tokens, latency_ms, $cost
```

## Sources

| Type | Resource |
|------|----------|
| Docs | [Claude Platform Docs — Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching) |
| Docs | [Claude Platform Docs — Model pricing & context windows](https://platform.claude.com/docs/en/about-claude/pricing) |
| Docs | [Claude Platform Docs — Choosing a model](https://platform.claude.com/docs/en/about-claude/models/choosing-a-model) |
| Course | DeepLearning.AI short courses on LLM application fundamentals (free) |

## Done when

You can predict, before running it, roughly what a given request will cost and how long it'll take — and defend a routing decision with numbers, not vibes.

---
[← Stage 01 — Python + Async Foundations](stage-01-python-async.md) · [Back to roadmap](../README.md) · **Next:** [Stage 03 — Tool Calling + Structured Outputs →](stage-03-tool-calling.md)
