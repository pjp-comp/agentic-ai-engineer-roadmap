[← Back to roadmap](../../README.md)

# Stage 02 — Model router (two-tier classification, instrumented)

[Stage 2's brief](../../docs/stage-02-llm-fundamentals.md#brief), made runnable:

> Build a router that classifies incoming requests by complexity (regex/heuristic first, cheap-model classifier second) and sends "simple" ones to a fast/cheap model, "complex" ones to a frontier model. Log tokens, latency, and $ per request for both paths.

## What it is

Six requests, a mix of trivial lookups and genuine reasoning tasks, run through a router that picks a model per request — then the same six forced through the expensive model, so the two totals can be compared.

**Ollama only by default, and free.** The two tiers are `llama3.2:3b` (cheap) and `llama3.1:8b` (frontier). Set `USE_LOCAL_MODEL=false` for `claude-haiku-4-5` / `claude-opus-5` instead; only the two model names change.

## Run it

```bash
ollama pull llama3.2:3b     # ~2GB, shared with most examples
ollama pull llama3.1:8b     # ~4.9GB, shared with stage05-multimodal

uv run agent.py                 # the 6-request workload, routed
uv run agent.py --no-route      # same workload, everything to the big model
uv run agent.py "your question" # route a single request
```

## The two-tier classifier is the actual lesson

The brief says "regex/heuristic first, cheap-model classifier second," and that ordering is the point:

| Tier | What it is | Cost | Handles |
|---|---|---|---|
| **1** | Keyword + length heuristic | Free, microseconds | Most traffic — the clear-cut cases |
| **2** | Ask the *small* model to classify | One cheap call | Only what tier 1 refused to guess at |

**If you put an LLM classifier in front of every request, you pay for two calls on every request to save money on some of them.** The heuristic exists so the classifier runs rarely. In the built-in workload, tier 1 settles 4 of 6 requests for free and tier 2 runs twice.

The second design choice worth copying: `classify_heuristic()` **returns `None` when it isn't sure**, rather than guessing. A heuristic that always produces an answer is just a bad classifier. Abstaining is what makes the escalation to tier 2 meaningful — and the case it abstains on is instructive:

```
"What is a vector database and why would an agent need one?"
  -> starts like a lookup ("what is"), ends like an analysis ("why")
  -> tier 1 sees both markers, abstains
  -> tier 2 correctly says COMPLEX
```

Note also that tier 2 runs on the **cheap** model. Paying frontier prices to decide whether to use the frontier model would defeat the whole exercise.

## The measured result — and why it's smaller than you'd expect

A real run of the six-request workload:

| | Routed | No routing |
|---|---|---|
| Answered by cheap model | 3 / 6 | 0 / 6 |
| Tier-2 classifier calls | 2 | 0 |
| **Total cost** | **$0.0397** | **$0.0443** |

**Routing saved about 10%.** That is a genuine, reproducible result, and it is deliberately not dressed up.

Why so modest? Because [Stage 2's cost table](../../docs/stage-02-llm-fundamentals.md#a-cost-reference-to-calibrate-against) already told you: the frontier-to-cheap spread is roughly 5×, not 100×, and **output tokens dominate**. The three genuinely complex requests each produced ~512 output tokens on the expensive model, and no amount of routing changes that — they *needed* that model. Routing only ever saves money on the requests that were overpaying, and here that was half of them, each one cheap to begin with.

Three things follow, and they're the real takeaways:

1. **Routing pays off in proportion to how much of your traffic is trivial.** A support bot answering the same ten questions all day saves enormously. A system where every request is a hard reasoning task saves nothing — and pays a classifier tax for the privilege.
2. **Measure before you build one.** This example took a few minutes to instrument and produced a number that argues *against* elaborate routing on this particular workload. That's the outcome the brief's "defend a routing decision with numbers, not vibes" is asking for.
3. **Fix the loop before you optimize the routing.** An agent making 20 calls where 3 would do wastes far more than a tier mismatch ever will. Routing is a last-10% optimization, not a first move.

## Reading the cost column

Local models cost nothing, which would make a cost column useless as a teaching device. So local runs report **real token counts** (from Ollama's `prompt_eval_count` / `eval_count`) priced at **published Claude rates**, labelled as simulated on every run. You watch what the routing would cost without spending anything.

Latency is real and unsimulated — and worth looking at, since it shows the other half of the tradeoff: the frontier model took ~33s per complex answer versus ~2s for a cheap one. Routing buys latency as well as cost, and on a user-facing path that often matters more.

## What to look at closely

- **`classify_heuristic()` returning `None`** — the abstain path, and the reason tier 2 exists at all.
- **`_COMPLEX_PATTERNS` matches task *shape*, not subject** — "compare", "design", "why" rather than topic keywords. Subject keywords would need endless maintenance; task-shape keywords stay valid as topics change.
- **`Call.cost` is computed, not estimated** — from real token counts and an explicit rate table, so the arithmetic is auditable.
- **`--no-route` exists at all** — a router with no baseline to compare against is an unfalsifiable claim. Always build the comparison.

## Extend it

1. Add a third tier (a mid model) and see whether it earns its place or just adds a classification decision that's wrong more often.
2. Log the classifier's verdict against your own judgement on 20 requests — it's [Stage 10's judge-validation idea](../../docs/stage-10-evaluation-qa.md#validate-the-judge-before-you-trust-the-metric) applied to the router. A miscalibrated router silently sends hard work to the cheap model.
3. Replace the static rate table with a cache-aware one and re-measure. [Prompt caching](../../docs/stage-02-llm-fundamentals.md#cost-optimization-beyond-routing) usually beats tier routing, which is why "cache first, route second" is the right order.

## Where this goes next

[Stage 2's multi-model section](../../docs/stage-02-llm-fundamentals.md#multi-model-agents--routing-across-vendors-not-just-tiers) takes the same `route()` function one step further — returning a *provider*, not just a model name. [`examples/stage02-basic-agent-local/`](../stage02-basic-agent-local/) is the minimal version of that split already: one flag, two backends, identical agent.

---
[← Back to roadmap](../../README.md) · [Stage 02 doc](../../docs/stage-02-llm-fundamentals.md)
