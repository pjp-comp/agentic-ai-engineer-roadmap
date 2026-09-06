[← Stage 01](stage-01-python-async.md) · Stage 02 of 14 · **Next:** [Stage 03 →](stage-03-tool-calling.md)

# Stage 02 — LLM Fundamentals for Agents

Context engineering · model routing · multi-model (cross-provider) agents · token economics · cost optimization · latency tradeoffs · failure modes

## Why this matters

Agent cost and latency are dominated by model choice and context size, not your orchestration code. Knowing when a small/fast model beats a frontier one — and how prompt caching changes the economics — is what separates a demo from something affordable to run at volume.

## How LLMs actually work

Every stage after this one talks about "tokens," "context windows," and "temperature" as if you already know what they are. This section is that missing prerequisite, covered once here so it doesn't need repeating.

**An LLM is a probabilistic next-token machine.** At each step, the model doesn't "decide" an answer — it computes a probability distribution over every possible next token given everything so far, then a sampling step picks one. That token gets appended to the input, and the whole process repeats. There's no separate "planning" phase happening before generation starts; even a model that appears to reason step-by-step is still predicting one token at a time, and what looks like reasoning is that same mechanism applied to text that happens to describe intermediate steps. This is *why* prompting works at all: you're not configuring a program, you're shaping the probability distribution the next prediction is drawn from.

**What is a token?** Not a word, and not a character — a token is a chunk of text the model's tokenizer maps to one integer ID, typically averaging around 3–4 characters of English text. "agentic" might be one token or split into pieces like "agent" + "ic" depending on the tokenizer's vocabulary; punctuation, whitespace, and non-English text often tokenize less efficiently than plain English words. This is the literal unit everything is billed and budgeted in — `tokens_in`/`tokens_out` in the brief below aren't an abstraction, they're a direct count of these chunks, and `MAX_ITERATIONS`/context-window limits throughout this roadmap are bounding a token count, not a word count or a turn count.

**Temperature, top-p, and other sampling parameters control how that next-token distribution gets sampled**, not what the model "knows":

| Parameter | What it does | Typical use |
|---|---|---|
| **Temperature** | Scales the probability distribution before sampling. Near 0: almost always picks the single highest-probability token (deterministic, repetitive). Higher (e.g. 0.7–1.0): flattens the distribution, so lower-probability tokens get picked more often (more varied, more prone to drift) | Low for tool-calling/structured extraction/classification (you want consistency); higher for brainstorming, creative writing, varied phrasing |
| **Top-p (nucleus sampling)** | Instead of considering every possible token, only samples from the smallest set of tokens whose cumulative probability exceeds p (e.g. 0.9). Often used instead of, or alongside, temperature | Narrowing the candidate pool without going fully deterministic |
| **Top-k** | Only considers the k most likely next tokens, discarding the rest before sampling | A blunter version of top-p; less commonly tuned directly in modern agent code |

You'll see `temperature=0.3` set directly in this repo's runnable examples (e.g. [`examples/stage04-memory-agent/agent.py`](../examples/stage04-memory-agent/agent.py)) — that's a low-but-not-zero setting chosen for a chat agent that should be consistent but not robotic. An agent calling tools with structured arguments generally wants temperature at or near 0: you're not looking for creative variation in a JSON schema.

**Runnable, hands-on version of this table:** [`examples/stage02-sampling-params/`](../examples/stage02-sampling-params/) — not an agent, just raw calls to a local model. Covers temperature/top-p side by side, a prompt's real token IDs plus token-by-token streamed generation, and a model's separate "thinking" field next to its final answer — so all of this section's claims are things you watch happen, not only read about.

## Core prompting technique, before context engineering replaces it

"Context engineering" (below) is the broader skill this stage is really teaching, but it's worth having the narrower prompting fundamentals first — context engineering is an extension of these ideas, not a replacement for needing them:

- **Be explicit about format and constraints**, don't rely on the model inferring what you want from a vague ask — "list 3 options as bullet points, no more than one sentence each" outperforms "give me some options."
- **Show, don't just tell (few-shot examples)** — one or two examples of the input/output shape you want, embedded in the prompt, does more to fix format drift than a paragraph of instructions describing the format in the abstract.
- **Give the model a role/persona when it changes behavior, not by default** — "You are a precise assistant" (used in this repo's `SYSTEM_PROMPT` constants) narrows tone and priorities; it's not required for every prompt, but it's cheap and effective when you need consistent framing across many calls.
- **Common pitfalls**: burying the actual instruction in the middle of a long prompt (models attend unevenly across long contexts — put the critical instruction near the start or end, not buried in the middle); assuming more instructions always help (conflicting or redundant instructions can *reduce* reliability, not improve it); and testing a prompt once and assuming it generalizes (temperature > 0 means the same prompt can produce different outputs across runs — a prompt that worked in one manual test can still fail intermittently).

"Thinking like an LLM" in practice means remembering the model has no access to anything outside the current context window and its training — if a fact, a prior decision, or a constraint isn't literally present in the tokens you send it, it doesn't exist for that call, no matter how "obvious" it seems to you. That single idea is also the seed of the next section.

## Context engineering — the discipline this stage is actually teaching

"Prompt engineering" means wording one instruction well. **Context engineering** is the broader, now-dominant skill: deciding *everything* that goes into the model's context window on a given call — which tool schemas, which memory, which retrieved chunks, how much conversation history, in what order — and, just as importantly, what to leave out. Model routing (the brief below) is context engineering applied to one axis: which model sees the context at all. Stages 3–6 are context engineering applied to the rest: which tools are visible (Stage 3), which memory tier answers a given question (Stage 4), which retrieved documents get included (Stage 5), which session state is in scope (Stage 6). None of those stages name it explicitly — this is the unifying concept behind all of them.

It's worth knowing by name because it's the term the field now uses for "why did the agent do something wrong" debugging: the answer is almost always "the context it saw was incomplete, stale, or contained the wrong things," not "the model reasoned badly."

### The four context failure modes — a debugging vocabulary

"The context was wrong" is too coarse to act on. These four named failures each have a *different* fix, and being able to tell them apart is the difference between guessing and debugging:

| Failure | What happens | Tell-tale sign | Fix |
|---|---|---|---|
| **Context poisoning** | A hallucination (or a bad tool result) enters the context and is then treated as established fact by every subsequent turn | The agent confidently repeats a specific wrong detail it invented several turns ago, and defends it | Validate tool output before it re-enters context; let a turn be *removed*, not just appended to. Stage 12's `wrap_as_untrusted_data` is this same idea applied to a hostile source instead of a mistaken one |
| **Context distraction** | The context grows so long the model over-attends to its own accumulated history and stops using what it actually knows | Performance degrades as a session gets longer, even though nothing is factually wrong in the history — the agent starts repeating past actions instead of doing new ones | Compaction/summarization past a threshold (Stage 4's fourth tier); the stagnation check in [Stage 7](stage-07-single-agent.md) catches the symptom, compaction addresses the cause |
| **Context confusion** | Irrelevant content — most often superfluous tool schemas — degrades the model's choices even when it's never used | Tool-selection accuracy drops as you add tools, and the agent calls plausible-but-wrong tools | Show only the tools relevant to the current task rather than every tool the system has; this is what Stage 3's dynamic tool discovery and the Skills pattern both exist to enable |
| **Context clash** | Two parts of the context contradict each other — an early turn says one thing, a later retrieval says another | Inconsistent answers to the same question within one session; the model appears to "flip-flop" | Supersession on write (Stage 4's forgetting policy) so a corrected fact replaces rather than accompanies the old one |

The reason these are worth memorizing: **three of the four get worse as the system gets more capable.** Adding more tools invites confusion, longer sessions invite distraction, and a bigger memory store invites clash. They're not beginner mistakes you grow out of — they're the failure modes that arrive *with* sophistication, which is why context engineering is a discipline and not a tip.

### Context budgets

The practical technique this adds on top of the brief: **context budgets**, not just model routing. Track tokens_in against a per-call ceiling the same way you'd track $cost, and when a long-running agent's context grows past a threshold, prune or compact it (drop old tool results, summarize stale turns) rather than letting it grow until the model truncates or the request errors — see [Stage 7's long-running-agent section](stage-07-single-agent.md#long-running-agents) for where this becomes unavoidable.

A useful framing for what you're actually doing with a budget: every token in the window is competing for the model's attention. The goal isn't "fit under the limit" — it's "maximize the share of the window that's relevant to the current step." A request at 40% of the context limit with 90% relevant content will outperform one at 80% of the limit with 30% relevant content, even though the second one technically "fits."

## Brief

Build a router that classifies incoming requests by complexity (regex/heuristic first, cheap-model classifier second) and sends "simple" ones to a fast/cheap model, "complex" ones to a frontier model. Log tokens, latency, and $ per request for both paths.

This cheap/fast-vs-frontier tiering pattern is universal, not Claude-specific — every major provider ships at least two tiers for exactly this tradeoff (OpenAI's GPT-5 mini vs. GPT-5, Gemini's Flash vs. Pro). The snippet below uses Claude's model names as this repo's concrete stand-in:

**Claude API example:**

```python
def route(task_complexity: str) -> str:
    return {
        "simple":  "claude-haiku-4-5",
        "complex": "claude-opus-5",
    }.get(task_complexity, "claude-sonnet-5")

# instrument every call:
# tokens_in, tokens_out, cache_read_tokens, latency_ms, $cost
```

## Multi-model agents — routing across vendors, not just tiers

The brief above routes within one vendor's tiers (Claude Haiku vs. Opus). **Multi-model** goes a level further: the router picks between entirely different providers or model families for the same task, not just cheap-vs-expensive versions of one. This repo's own `USE_LOCAL_MODEL` examples (`examples/stage03-tool-calling/`, `examples/stage02-basic-agent-local/`, `examples/stage04-*`) are a working instance of this — one flag switches the exact same agent between Claude and a local Llama model, with the rest of the code untouched.

Reasons a real system reaches for this, beyond the single-vendor router's cost/latency tradeoff:

- **Cost floor** — a local open-weight model has zero marginal cost per call once downloaded; routing high-volume, low-stakes calls there (not just to a cheaper *tier* of the same paid API) can eliminate cost entirely for that slice of traffic, not just reduce it.
- **Availability/fallback** — if one provider has an outage or rate-limits you, a router that only knows one vendor's model names has no fallback at all. A multi-model router can fail over to a different provider, not just a different tier of a now-unavailable one.
- **Avoiding lock-in** — a codebase that only ever calls one vendor's SDK makes switching providers later a rewrite, not a config change. The `_build_llm()` pattern in this repo's examples (branch on a flag, return an object satisfying the same interface) is the concrete shape of avoiding that.
- **Playing to model-specific strengths** — some models are measurably better at specific sub-tasks (e.g. one family's tool-calling reliability vs. another's summarization quality); a router can send each sub-task to whichever model is actually best at it, not just whichever is cheapest.

The mechanics are the same `route()` function from the brief above — it just returns a *provider* along with (or instead of) a model name, and the calling code needs a `_build_llm()`-style branch (as in this repo's examples) rather than a single hardcoded SDK client. The real cost isn't the routing logic — it's that every downstream assumption (tool-call schema shape, `stop_reason` vs. `finish_reason`, streaming format) now has to be normalized across providers, which is exactly the kind of translation work `examples/stage03-mcp-tool-server/agent.py`'s dual `_mcp_tool_to_claude_schema`/`_mcp_tool_to_ollama_schema` functions show concretely.

**A caveat worth knowing before reaching for this by default**: mixing model families isn't free of behavioral risk. Different models don't just vary in cost and speed — they vary in how strictly they follow tool schemas (see the `stage03-multi-tool-agent` note above on Llama sending a number as a string where Claude didn't) and in how reliably they judge *when* to call a tool at all (see `examples/stage04-longterm-memory-agent/README.md`'s documented gap). A multi-model router needs the same defensive input handling and evals (Stage 10) applied per-provider, not assumed to transfer from whichever model you tested with first.

At production scale, a gateway (e.g. LiteLLM, already cited in [Stage 13](stage-13-deployment.md)) normalizes this translation work across providers for you, rather than hand-writing `_build_llm()`-style branches per project — worth reaching for once you have more than one or two providers to route across.

## Sources

| Type | Resource |
|------|----------|
| Docs | [Claude Platform Docs — Token counting](https://platform.claude.com/docs/en/build-with-claude/token-counting) — what a token actually is, how to count them before sending a request |
| Docs | [Hugging Face — Summary of the tokenizers](https://huggingface.co/docs/transformers/en/tokenizer_summary) — vendor-neutral explanation of how text becomes tokens (BPE and related schemes) |
| Docs | [OpenAI — Text generation, incl. temperature/top-p](https://developers.openai.com/api/docs/guides/text) — sampling parameters explained from a second provider, worth comparing against Claude's `temperature`/`top_p` in the [Messages API reference](https://platform.claude.com/docs/en/api/messages) |
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

## Cost optimization beyond routing

Model routing is the first lever, not the only one. Once a router is in place, these compound with it:

- **Cache-hit-rate design** — prompt caching (already in the Sources below) only pays off if the cached prefix is stable across calls. Put static content (system prompt, tool definitions) first and volatile content (the user's actual message) last, so the same prefix is reused call after call instead of invalidating the cache every time.
- **Batch-tier processing** — for anything that doesn't need a synchronous response (nightly scoring, bulk classification, eval runs), batch APIs typically cut cost roughly in half in exchange for async turnaround. Not usable for interactive agent turns, but often applicable to Stage 10's eval suite itself.
- **Per-session token budgets** — cap total tokens a single agent run is allowed to consume, enforced at the call site (or a gateway in front of it), independent of the iteration cap from Stage 7. An agent that's technically not looping forever can still be quietly expensive per run without one.

## A cost reference to calibrate against

This stage's "Done when" asks you to predict a request's cost before running it. That's impossible without knowing roughly what the tiers cost, so here's a reference point — **prices drift, so treat the shape as the lesson and re-check the numbers** at [Claude](https://platform.claude.com/docs/en/about-claude/pricing) / [OpenAI](https://developers.openai.com/api/docs/pricing) before quoting them anywhere that matters:

| Tier | Example | Input $/1M tokens | Output $/1M tokens |
|---|---|---|---|
| **Local / open-weight** | Llama 3.2 3B via Ollama (what this repo's examples use) | $0 marginal | $0 marginal |
| **Small / fast** | Claude Haiku 4.5 | ~$1 | ~$5 |
| **Mid** | Claude Sonnet 5 | ~$2 | ~$10 |
| **Frontier** | Claude Opus 5 | ~$5 | ~$25 |

Three things worth internalizing from the shape of that table, none of which are obvious until you've been billed by one of them:

1. **Output costs ~5× input.** An agent that returns verbose answers is much more expensive than one that reads a lot and answers concisely. Trimming a system prompt saves less than trimming the response length.
2. **The frontier-to-cheap spread is ~5×, not 100×.** Routing is worth doing, but it isn't the difference between viable and unviable — a badly-designed loop that makes 20 calls where 3 would do costs far more than picking the "wrong" tier. Fix the loop before you optimize the routing.
3. **Prompt caching changes the math more than model choice does.** A cached input token typically costs a fraction of a fresh one, so a stable prefix across many calls (Stage 2's cache-hit-rate design above) can beat a tier downgrade *without* the quality cost — which is why "cache first, route second" is the right order.

**Working an example.** A ReAct agent with a 2,000-token system prompt + tool schemas, running 5 loop iterations on a task, re-sending the growing history each time, and producing 300 output tokens per step: input is roughly 2,000 + (2,000 + accumulated results) × 4 ≈ 15,000 tokens; output is ~1,500. On a mid-tier model that's roughly $0.03 + $0.015 ≈ **$0.045 per task run.** Cheap for one run; $450 at 10,000 runs a day — which is exactly the point at which Stage 8's "multi-agent costs ~3× a single agent" warning stops being an abstraction. Do this arithmetic once for your own agent and the rest of this roadmap's cost advice becomes concrete.

---
[← Stage 01 — Python + Async Foundations](stage-01-python-async.md) · [Back to roadmap](../README.md) · **Next:** [Stage 03 — Tool Calling + Structured Outputs →](stage-03-tool-calling.md)
