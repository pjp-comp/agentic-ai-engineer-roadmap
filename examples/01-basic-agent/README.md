[← Back to roadmap](../../README.md)

# Example 01 — A Basic AI Agent

This is the smallest thing that's honestly an *agent* and not just an API call: the model can decide, on its own, to call a tool, see the result, and decide again — until it's ready to answer. No LangGraph, no CrewAI — just a raw SDK call and a `while` loop, so the loop itself isn't a black box before you start using frameworks that hide it.

**Runs against a free local Ollama model by default** (`USE_LOCAL_MODEL=true`, both in `.env.example` and as `agent.py`'s own fallback) — no API key needed to try it. Set `USE_LOCAL_MODEL=false` for the Claude API instead; both paths use the exact same loop and tool.

## What it is

- **One tool**: a calculator (`calculate`), implemented with a safe AST-based evaluator — not `eval()`.
- **One loop**: send messages → if the model asks for a tool, run it and send the result back → repeat, capped at `MAX_ITERATIONS` so it can't run forever.
- **~120 lines total.** Read `agent.py` top to bottom; nothing in it is hidden.

This is an **AI agent**, not agentic AI — see the [distinction in the main README](../../README.md#ai-agent-vs-agentic-ai--the-distinction-that-matters). It has one role, one bounded task, and stops when it has an answer. It does not plan, delegate, or pursue a goal across multiple sessions.

## Run it

Managed with [`uv`](https://docs.astral.sh/uv/) — `uv run` creates an isolated virtualenv, installs the locked dependencies, and runs the script in one step, so there's no separate "activate your venv" ritual.

Copy `.env.example` to `.env` once, at the repo root — every example picks it up automatically:

```bash
cp ../../.env.example ../../.env   # from this directory, or just edit the repo-root .env.example
```

**Local (default, free):**

```bash
ollama pull llama3.2:3b   # one-time, ~2GB
cd examples/01-basic-agent
uv run agent.py "What is 23 * 47, plus 100, all divided by 3?"
```

**Claude API instead:** set `USE_LOCAL_MODEL=false` and `ANTHROPIC_API_KEY=sk-ant-...` in `.env`, then run the same command.

The first run downloads dependencies into a local `.venv/` (git-ignored) per the pinned versions in `uv.lock`; every run after that is instant. No `pip install` step, no manual venv activation.

`agent.py` loads `../../.env` via `python-dotenv` at startup — no `export` needed in your shell. Exporting the same variables in your shell still works and takes precedence if both are set.

<details>
<summary>Without <code>uv</code> (plain <code>pip</code>)</summary>

```bash
cd examples/01-basic-agent
python -m venv .venv && source .venv/bin/activate
pip install anthropic ollama python-dotenv
python agent.py "What is 23 * 47, plus 100, all divided by 3?"   # local by default
```
</details>

Expected output (tool calls print to stderr so stdout stays clean):

```
  [tool call] calculate({'expression': '23 * 47'})
  [tool call] calculate({'expression': '1081 + 100'})
  [tool call] calculate({'expression': '1181 / 3'})
393.6666666666667
```

Try it without a math question too (`uv run agent.py "What's the capital of France?"`) — the model should answer directly, with zero tool calls, since `stop_reason` never becomes `"tool_use"`.

## What to look at closely

- **`messages.append({"role": "assistant", "content": response.content})`** — the full content list is appended, not just the text. Drop the `tool_use` blocks here and the next request breaks, because a `tool_result` with no matching `tool_use_id` in history is invalid.
- **`tool_results` collected before appending** — if a turn requests multiple tools, all their results go back in a *single* user message. Splitting them across messages silently trains the model to stop batching tool calls.
- **`MAX_ITERATIONS`** — this is the "graceful degradation" concept from [Stage 7](../../docs/stage-07-single-agent.md): an agent that can loop forever on a flaky tool is a liability, not a feature.
- **AST whitelist, not `eval()`** — the model's tool input is untrusted text. `calculate()` only ever evaluates arithmetic node types; anything else raises before it's touched.
- **The local path is deliberately simple, and can give a wrong answer** — `_run_agent_local` is structurally identical to `_run_agent_claude`: check `tool_calls`, run the tool, or return the text, nothing else. On a genuinely multi-step task, Llama 3.2 3B doesn't always get it right — it can build the wrong expression for an ambiguous word problem (e.g. `20*40+2+3/2` = `803.5` instead of the intended `(20*40+2+3)/2` = `402.5`), or answer with its own mental math instead of a second tool call. `calculate()` will always compute *whatever expression it's given* correctly — a wrong answer here means the model chose the wrong expression or skipped the tool, not that the calculator is broken. Try `uv run agent.py "multiply twenty and forty then add 2 and 3, then divide the result by 2"` a handful of times to see the real range of behavior, wrong answers included. An earlier version of this file tried to detect and auto-correct this in the loop itself; that made the code more complex without reliably fixing anything, so it was removed — the system prompt's explicit "parenthesize the result of an earlier step" instruction is the one improvement that's still here, and it measurably helps without adding branching logic to the loop. If getting the *right* answer matters more than watching the tool-calling mechanics, ask for the arithmetic in already-unambiguous form (`(20*40+2+3)/2`) or switch to the Claude path (`USE_LOCAL_MODEL=false`), which doesn't have this failure mode — `tool_use` is a structured response field there, not something the model can approximate in prose.

## Extend it (optional exercises)

1. Add a second tool (e.g. a `word_count` tool) and watch the model choose between them based on the task — or see it already done in [`examples/01_multi_tool_agent/`](../01_multi_tool_agent/), which adds four tools and a dispatch table.
2. Force a tool error (e.g. `calculate("1/0")`) and confirm the model recovers instead of crashing — this is the `is_error` pattern from [Stage 3](../../docs/stage-03-tool-calling.md).
3. Add a hard iteration cap test: ask a question that can't be answered with the calculator tool and confirm the loop still terminates cleanly via `MAX_ITERATIONS`.

## Where this goes next

This single agent is the building block for **agentic AI** (Stage 8): once you have one agent that reliably loops, the next step is wiring two of them together — e.g. a "planner" agent that breaks a task into sub-tasks and a "worker" agent (this same loop) that executes each one, coordinated by a supervisor. Stage 6's shared-session pattern is how they'd actually communicate. That two-agent handoff is a natural `examples/02-two-agent-handoff/` to build once Stages 6–8 in the main roadmap feel solid — not included yet, since it's meant to be *your* next exercise, not another script to read.
