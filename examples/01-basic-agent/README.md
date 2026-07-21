[← Back to roadmap](../../README.md)

# Example 01 — A Basic AI Agent

This is the smallest thing that's honestly an *agent* and not just an API call: the model can decide, on its own, to call a tool, see the result, and decide again — until it's ready to answer. No LangGraph, no CrewAI — just the Claude API and a `while` loop, so the loop itself isn't a black box before you start using frameworks that hide it.

## What it is

- **One tool**: a calculator (`calculate`), implemented with a safe AST-based evaluator — not `eval()`.
- **One loop**: send messages → if the model asks for a tool, run it and send the result back → repeat, capped at `MAX_ITERATIONS` so it can't run forever.
- **~120 lines total.** Read `agent.py` top to bottom; nothing in it is hidden.

This is an **AI agent**, not agentic AI — see the [distinction in the main README](../../README.md#ai-agent-vs-agentic-ai--the-distinction-that-matters). It has one role, one bounded task, and stops when it has an answer. It does not plan, delegate, or pursue a goal across multiple sessions.

## Run it

```bash
cd examples/01-basic-agent
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
python agent.py "What is 23 * 47, plus 100, all divided by 3?"
```

Expected output (tool calls print to stderr so stdout stays clean):

```
  [tool call] calculate({'expression': '23 * 47'})
  [tool call] calculate({'expression': '1081 + 100'})
  [tool call] calculate({'expression': '1181 / 3'})
393.6666666666667
```

Try it without a math question too (`python agent.py "What's the capital of France?"`) — the model should answer directly, with zero tool calls, since `stop_reason` never becomes `"tool_use"`.

## What to look at closely

- **`messages.append({"role": "assistant", "content": response.content})`** — the full content list is appended, not just the text. Drop the `tool_use` blocks here and the next request breaks, because a `tool_result` with no matching `tool_use_id` in history is invalid.
- **`tool_results` collected before appending** — if a turn requests multiple tools, all their results go back in a *single* user message. Splitting them across messages silently trains the model to stop batching tool calls.
- **`MAX_ITERATIONS`** — this is the "graceful degradation" concept from [Stage 5](../../docs/stage-05-single-agent.md): an agent that can loop forever on a flaky tool is a liability, not a feature.
- **AST whitelist, not `eval()`** — the model's tool input is untrusted text. `calculate()` only ever evaluates arithmetic node types; anything else raises before it's touched.

## Extend it (optional exercises)

1. Add a second tool (e.g. a `word_count` tool) and watch the model choose between them based on the task.
2. Force a tool error (e.g. `calculate("1/0")`) and confirm the model recovers instead of crashing — this is the `is_error` pattern from [Stage 3](../../docs/stage-03-tool-calling.md).
3. Add a hard iteration cap test: ask a question that can't be answered with the calculator tool and confirm the loop still terminates cleanly via `MAX_ITERATIONS`.

## Where this goes next

This single agent is the building block for **agentic AI** (Stage 6): once you have one agent that reliably loops, the next step is wiring two of them together — e.g. a "planner" agent that breaks a task into sub-tasks and a "worker" agent (this same loop) that executes each one, coordinated by a supervisor. That two-agent handoff is a natural `examples/02-two-agent-handoff/` to build once Stage 5 and Stage 6 in the main roadmap feel solid — not included yet, since it's meant to be *your* next exercise, not another script to read.
