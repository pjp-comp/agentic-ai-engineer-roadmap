[← Back to roadmap](../../README.md)

# Example 00 — Sampling Parameters (Not an Agent)

Every other example in this repo is an agent: tools, a loop, a decision point. This one deliberately isn't — it's the raw mechanism underneath all of them, made hands-on instead of just described in [Stage 2's sampling-parameters table](../../docs/stage-02-llm-fundamentals.md#how-llms-actually-work--the-mechanics-everything-else-in-this-roadmap-assumes). Every `agent.py` in this repo sets `temperature=0` (or `temperature=0.3`) without explaining why — this is where that "why" becomes something you watch happen, before any tool-calling or loop complexity gets added on top.

## What it is

- **One function, `ask()`** — a single raw call to a local Ollama model. No tools, no loop, no `TOOLS` list. This *is* the whole "agent" — send a prompt, get text back.
- **One model for everything: `qwen3:1.7b`** (~1.4GB) — unlike `llama3.2:3b` used elsewhere in this repo, it supports "thinking" mode, so the same model covers every subcommand below with nothing extra to pull.
- **Five subcommands**, each isolating one thing to observe:
  - `same-prompt` — run one prompt N times at a fixed setting, to see run-to-run variance at that setting alone.
  - `compare-temps` — run one prompt at temperature 0, 0.5, and 1.0, side by side.
  - `compare-top-p` — run one prompt at top_p 0.1, 0.5, and 1.0, with temperature held fixed at 0.9 so top_p's effect isn't confounded with temperature's.
  - `tokenize` — show a prompt's real token IDs (this model's actual tokenizer, via Ollama's `context` field) and then stream generation one token at a time, so a token visibly isn't a word or a character.
  - `thinking` — show the model's separate `thinking` field (its reasoning) next to its `content` field (the actual answer) — two genuinely different fields in one API response.
- **Local-only, on purpose** — Ollama's `chat()`/`generate()` expose `temperature`/`top_p`/`top_k`/`think` directly via kwargs and an `options` dict, and this is a parameter-mechanics lesson, not a provider comparison. No Claude path here.

## Run it

```bash
cd examples/00_llm_fundamentals
ollama pull qwen3:1.7b   # one-time, ~1.4GB — the only model this file needs
uv run sampling_params.py compare-temps
```

Expected shape of the output — same prompt, three temperatures, increasingly varied phrasing:

```
--- temperature=0.0 ---
Elephants can hear sounds up to 160 decibels, making them one of the most sensitive hearing animals in the world.

--- temperature=0.5 ---
Elephants can remember their childhood friends for decades, showing remarkable social intelligence.

--- temperature=1.0 ---
Elephants can understand human speech when exposed to it, though they can only comprehend simple words and not full sentences.
```

Try the rest:

```bash
uv run sampling_params.py same-prompt --temperature 0 --runs 3      # should print the same answer 3 times
uv run sampling_params.py same-prompt --temperature 1.0 --runs 3    # should print 3 different answers
uv run sampling_params.py compare-top-p
uv run sampling_params.py tokenize --prompt "The quick brown fox jumps"
uv run sampling_params.py thinking --prompt "What is 12 + 7?"
```

Use your own prompt with any subcommand:

```bash
uv run sampling_params.py same-prompt --prompt "Name a color." --temperature 1.0 --runs 3
```

## What to look at closely

- **`temperature=0` is deterministic, or very close to it** — `same-prompt --temperature 0 --runs 3` should print the identical answer three times. This is *why* every tool-calling `agent.py` in this repo defaults to `temperature=0`: you want the same input to reliably produce the same tool call, not creative variation in a JSON schema.
- **`temperature=1.0` doesn't always look different** — try `same-prompt --prompt "Name a color." --temperature 1.0 --runs 3`. Depending on the prompt, the model's probability distribution can be so heavily skewed toward one answer that sampling variety barely registers even at high temperature. Temperature doesn't force variety — it makes lower-probability tokens *eligible*, but if one token dominates the distribution enough, it still wins most of the time. This is honest, useful signal about how skewed a distribution actually is, not a bug in the script.
- **`compare-top-p` holds temperature fixed at 0.9** — deliberately, so what you're watching is top_p's effect in isolation. top_p narrows the *candidate pool* before sampling (only tokens whose cumulative probability reaches p are eligible at all); temperature reshapes the *probabilities* within whatever pool ends up being sampled from. They're two different knobs on the same distribution, not two names for the same thing — this is the distinction [Stage 2's table](../../docs/stage-02-llm-fundamentals.md) describes and this script lets you actually see.
- **`think=False` is explicit everywhere except `thinking`** — `qwen3:1.7b` defaults to thinking mode ON, even without asking for it. Without `think=False`, every temperature/top-p/tokenize demo would silently include an invisible reasoning pass, confounding what you're actually measuring (worse: `tokenize`'s 15-token generation budget can get entirely consumed by hidden thinking, leaving an empty visible response). `thinking` flips this deliberately — `think=True` — since seeing that field populated is the entire point of that subcommand.
- **`tokenize`'s token count is larger than the word count** — and includes IDs that aren't part of what you typed at all: those are the model's own chat template (role markers, structural tokens Ollama adds automatically). This is the literal unit `tokens_in` cost is billed in — not words, not characters, not even only the text you wrote. The streamed generation afterward shows the same thing from the output side: each printed chunk is one real token, and they don't line up cleanly with word boundaries.
- **`thinking` and `content` are separate fields, not the same text twice** — a model without thinking support (like `llama3.2:3b`, used everywhere else in this repo) only ever produces `content`; there's no hidden reasoning step you're just not seeing; the mechanism genuinely isn't there for that model. Thinking tokens also count against your token budget even though they're never the visible answer — worth knowing before assuming a longer visible response is the only place cost comes from.
- **`ask()` is the entire function** — no `MAX_ITERATIONS`, no tool schema, no message-history management. Comparing this file's simplicity to [`examples/01-basic-agent/agent.py`](../01-basic-agent/agent.py)'s ~120 lines is itself the lesson: everything Stage 3 onward adds (tools, loops, memory) is built on top of this one function call, not a replacement for it.

## Where this goes next

Read [Stage 2's full section](../../docs/stage-02-llm-fundamentals.md#how-llms-actually-work--the-mechanics-everything-else-in-this-roadmap-assumes) for the token/next-token-prediction fundamentals this script assumes, then move to [`examples/01-basic-agent/`](../01-basic-agent/) — the same `temperature=0` philosophy demonstrated here is exactly why that example's tool-calling loop is reliable across repeated runs.
