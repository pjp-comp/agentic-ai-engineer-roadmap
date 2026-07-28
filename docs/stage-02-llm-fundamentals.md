[← Stage 01](stage-01-python-async.md) · Stage 02 of 14 · **Next:** [Stage 03 →](stage-03-tool-calling.md)

# Stage 02 — LLM Fundamentals for Agents

Context engineering · model routing · token economics · cost optimization · latency tradeoffs · failure modes

## Why this matters

Agent cost and latency are dominated by model choice and context size, not your orchestration code. Knowing when a small/fast model beats a frontier one — and how prompt caching changes the economics — is what separates a demo from something affordable to run at volume.

## Brief

Build a router that classifies incoming requests by complexity (regex/heuristic first, cheap-model classifier second) and sends "simple" ones to a fast/cheap model, "complex" ones to a frontier model. Log tokens, latency, and $ per request for both paths.

This cheap/fast-vs-frontier tiering pattern is universal, not Claude-specific — every major provider ships at least two tiers for exactly this tradeoff (OpenAI's GPT-5 mini vs. GPT-5, Gemini's Flash vs. Pro). The snippet below uses Claude's model names as this repo's concrete stand-in:

**Claude API example:**

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
| Docs | [OpenAI — Prompt caching](https://developers.openai.com/api/docs/guides/prompt-caching) — same concept, different provider; worth comparing cache-write/read pricing and prefix rules against Claude's |
| Docs | [Claude Platform Docs — Model pricing & context windows](https://platform.claude.com/docs/en/about-claude/pricing) |
| Docs | [OpenAI — API pricing](https://developers.openai.com/api/docs/pricing) — cross-provider pricing and model-tier comparison |
| Docs | [Claude Platform Docs — Choosing a model](https://platform.claude.com/docs/en/about-claude/models/choosing-a-model) |
| Course | DeepLearning.AI short courses on LLM application fundamentals (free) |
| Guide | [Sourcegraph — Context Engineering Guide](https://sourcegraph.com/blog/context-engineering) — why this superseded "prompt engineering" as the primary practitioner skill |
| Guide | [Neo4j — Context Engineering vs. Prompt Engineering](https://neo4j.com/blog/agentic-ai/context-engineering-vs-prompt-engineering/) |
| Guide | [Galileo — Caching Playbook for Agents](https://galileo.ai/blog/the-2026-caching-playbook-for-agents-bigger-prompts-smaller-bills) — cache-prefix ordering for hit-rate |
| Guide | [Zylos — Token Budgets, Model Routing, FinOps for Agents](https://zylos.ai/research/2026-04-12-ai-agent-cost-optimization-token-budget-model-routing/) |

## Done when

You can predict, before running it, roughly what a given request will cost and how long it'll take — and defend a routing decision with numbers, not vibes.

## Context engineering — the discipline this stage is actually teaching

"Prompt engineering" means wording one instruction well. **Context engineering** is the broader, now-dominant skill: deciding *everything* that goes into the model's context window on a given call — which tool schemas, which memory, which retrieved chunks, how much conversation history, in what order — and, just as importantly, what to leave out. Model routing (the brief above) is context engineering applied to one axis: which model sees the context at all. Stages 3–6 are context engineering applied to the rest: which tools are visible (Stage 3), which memory tier answers a given question (Stage 4), which retrieved documents get included (Stage 5), which session state is in scope (Stage 6). None of those stages name it explicitly — this is the unifying concept behind all of them, and it's worth knowing by name because it's the term the field now uses for "why did the agent do something wrong" debugging: the answer is almost always "the context it saw was incomplete, stale, or contained the wrong things," not "the model reasoned badly."

The practical technique this adds on top of the brief: **context budgets**, not just model routing. Track tokens_in against a per-call ceiling the same way you'd track $cost, and when a long-running agent's context grows past a threshold, prune or compact it (drop old tool results, summarize stale turns) rather than letting it grow until the model truncates or the request errors — see [Stage 7's long-running-agent section](stage-07-single-agent.md#long-running-agents) for where this becomes unavoidable.

## Cost optimization beyond routing

Model routing is the first lever, not the only one. Once a router is in place, these compound with it:

- **Cache-hit-rate design** — prompt caching (already in the Sources below) only pays off if the cached prefix is stable across calls. Put static content (system prompt, tool definitions) first and volatile content (the user's actual message) last, so the same prefix is reused call after call instead of invalidating the cache every time.
- **Batch-tier processing** — for anything that doesn't need a synchronous response (nightly scoring, bulk classification, eval runs), batch APIs typically cut cost roughly in half in exchange for async turnaround. Not usable for interactive agent turns, but often applicable to Stage 10's eval suite itself.
- **Per-session token budgets** — cap total tokens a single agent run is allowed to consume, enforced at the call site (or a gateway in front of it), independent of the iteration cap from Stage 7. An agent that's technically not looping forever can still be quietly expensive per run without one.

## Sources
