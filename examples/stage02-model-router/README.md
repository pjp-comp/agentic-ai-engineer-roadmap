[← Back to roadmap](../../README.md)

# Stage 02 — Model router (two-tier classification, instrumented)

[Stage 2's brief](../../docs/stage-02-llm-fundamentals.md#brief), made runnable:

> Build a router that classifies incoming requests by complexity (regex/heuristic first, cheap-model classifier second) and sends "simple" ones to a fast/cheap model, "complex" ones to a frontier model. Log tokens, latency, and $ per request for both paths.

## What it is

Six requests, a mix of trivial lookups and genuine reasoning tasks, run through a router that picks a model per request — then the same six forced through the expensive model, so the two totals can be compared.

**Ollama only by default, and free.** The two tiers are `llama3.2:3b` (cheap) and `llama3.1:8b` (frontier). Set `USE_LOCAL_MODEL=false` for `claude-haiku-4-5` and a model ID you set in `ANTHROPIC_LARGE_MODEL`. Configure the four rate variables in `.env.example` from your provider's current rates.

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

## Measure your own result

Run the same workload with routing and with `--no-route`. Compare answer quality, classifier calls, input/output tokens, latency, and estimated cost. The larger local model is just the more capable tier in this exercise, not a claim of frontier capability.

The default rate table is illustrative. Local inference has no hosted token charge but consumes compute; pricing its measured tokens at those rates is a simulation. For hosted runs, set `SMALL_INPUT_RATE`, `SMALL_OUTPUT_RATE`, `LARGE_INPUT_RATE`, and `LARGE_OUTPUT_RATE` from the provider's current pricing. The simple formula omits caching and tool fees. See [cost arithmetic](../../docs/stage-02-llm-fundamentals.md#calculate-cost-from-measured-usage).

There is no guaranteed saving: a classifier adds work, output lengths vary, and routing mistakes can reduce quality. Keep a baseline and repeat measurements rather than presenting a single historical run as reproducible performance.

## What to look at closely

- **`classify_heuristic()` returning `None`** — the abstain path, and the reason tier 2 exists at all.
- **`_COMPLEX_PATTERNS` matches task *shape*, not subject** — "compare", "design", "why" rather than topic keywords. Subject keywords would need endless maintenance; task-shape keywords stay valid as topics change.
- **`Call.cost` is an estimate computed from token counts** — from real token counts and an explicit rate table, so the arithmetic is auditable.
- **`--no-route` exists at all** — a router with no baseline to compare against is an unfalsifiable claim. Always build the comparison.

## Extend it

1. Add a third tier (a mid model) and see whether it earns its place or just adds a classification decision that's wrong more often.
2. Log the classifier's verdict against your own judgement on 20 requests — it's [Stage 10's judge-validation idea](../../docs/stage-10-evaluation-qa.md#validate-the-judge-before-you-trust-the-metric) applied to the router. A miscalibrated router silently sends hard work to the cheap model.
3. Replace the static rate table with a cache-aware one and re-measure. Compare [prompt caching](../../docs/stage-02-llm-fundamentals.md#cost-optimization-beyond-routing) and routing on the same workload.

## Where this goes next

[Stage 2's multi-model section](../../docs/stage-02-llm-fundamentals.md#multi-model-agents--routing-across-vendors-not-just-tiers) takes the same `route()` function one step further — returning a *provider*, not just a model name. [`examples/stage02-basic-agent-local/`](../stage02-basic-agent-local/) is the minimal version of that split already: one flag, two backends, identical agent.

---
[← Back to roadmap](../../README.md) · [Stage 02 doc](../../docs/stage-02-llm-fundamentals.md)
