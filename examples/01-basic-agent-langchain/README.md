[← Back to roadmap](../../README.md)

# Example 01c — The Same Agent, Using LangChain (No LangGraph)

Same calculator, same task, same iteration cap as [`examples/01-basic-agent/`](../01-basic-agent/) and [`examples/01-basic-agent-langgraph/`](../01-basic-agent-langgraph/). This one exists to answer a question that comes up constantly: **isn't LangChain enough on its own — why do I need LangGraph too?**

The honest answer is in this file's loop: **you still have to write it by hand.** LangChain gives you a nicer model wrapper (`ChatAnthropic`) and automatic tool-schema generation (`@tool`), but it has no primitive for "call the model, then if it asked for a tool, run it and call the model again." That's a *cycle* — LangChain's abstractions (chains, `prompt | llm | parser`) are for linear or lightly-branching pipelines, not loops. Open `agent.py` and you'll find a `for step in range(MAX_ITERATIONS):` loop that is structurally identical to example 01's — same shape, just LangChain's classes instead of the raw Anthropic SDK's.

## The three-way comparison

| | [`01-basic-agent`](../01-basic-agent/) | `01-basic-agent-langchain` (this one) | [`01-basic-agent-langgraph`](../01-basic-agent-langgraph/) |
|---|---|---|---|
| Model call | `anthropic.Anthropic().messages.create(...)` | `ChatAnthropic(...).invoke(...)` | `ChatAnthropic(...).invoke(...)` |
| Tool schema | Hand-written `input_schema` JSON dict | `@tool` decorator (auto-derived) | `@tool` decorator (auto-derived) |
| **The loop itself** | **Hand-written `for` loop** | **Hand-written `for` loop** | **`StateGraph` with a conditional edge** |
| Tool dispatch | Hand-written `if block.type == "tool_use": ...` | Hand-written `if tool_fn: ... else: ...` | `ToolNode(TOOLS)` (automatic) |
| "Are we done?" check | `if response.stop_reason != "tool_use"` | `if not response.tool_calls` | `tools_condition` (a graph edge) |

Read the middle column: it has the *nicer model wrapper and schema generation* from LangChain, but everything about *looping and dispatching tool calls* is exactly as hand-written as example 01. That's the real dividing line:

- **LangChain** = a library of components (model wrappers, prompt templates, tool schemas, output parsers). Good for pipelines: input → prompt → LLM → parse → output, no going back.
- **LangGraph** = a graph/state engine, built on top of LangChain's components, that adds the thing LangChain doesn't have: cycles, conditional routing, and state that persists across steps. This is what actually replaces the hand-written loop — not LangChain alone.

If you only ever needed a single LLM call with no tool round-trip, LangChain alone would be enough and this comparison wouldn't matter. The moment a task needs "maybe call a tool, then decide what to do next based on the result" — which is the definition of an agent — you need either a hand-written loop (this file, or example 01) or LangGraph (which writes the loop for you as graph edges).

## Run it

```bash
cd examples/01-basic-agent-langchain
uv run agent.py "What is 23 * 47, plus 100, all divided by 3?"
```

Uses the same repo-root `.env` as the other two examples (see [`.env.example`](../../.env.example)) via `python-dotenv` — no `export` needed.

Expected output (tool calls print to stderr so stdout stays clean):

```
  [tool call] calculate({'expression': '(23 * 47 + 100) / 3'})
The result is **393.67** (approximately, or exactly 393⅔).

Here's the breakdown:
- 23 × 47 = 1081
- 1081 + 100 = 1181
- 1181 ÷ 3 = 393.67
```

Try `uv run agent.py "What's the capital of France?"` too — `response.tool_calls` is empty, the loop exits on the first iteration, same as the other two examples' no-tool-call path.

## What to look at closely

- **The `for step in range(MAX_ITERATIONS):` loop** — this is the whole point of this example. Compare it side by side with example 01's loop and example 01-langgraph's graph. LangChain got you `ChatAnthropic` and `@tool`; it did not get you out of writing this loop.
- **`messages.append(response)` then `messages.append(ToolMessage(...))`** — LangChain models the conversation as a flat list of typed message objects (`HumanMessage`, `AIMessage`, `ToolMessage`), one `ToolMessage` per tool call, instead of Anthropic's raw API shape (one `tool_result` content block per call, batched into a single user message). Same information, different object model.
- **`llm.bind_tools(TOOLS)`** — identical call to the one in the LangGraph example; this part of LangChain is shared by both.
- **No `StateGraph`, no `ToolNode`, no `tools_condition` anywhere in this file** — that's deliberate, not an oversight. Grep for them if you don't believe it.

## Where this goes next

This example's entire purpose is contrast, not a recommended path forward. For a real project, once you need looping/tool-calling agents (which is most agents), go straight to [`01-basic-agent-langgraph`](../01-basic-agent-langgraph/) and [Stage 8](../../docs/stage-08-multi-agent.md) — there's little reason to hand-write this loop with LangChain's classes when LangGraph expresses it more clearly as graph edges. This example is here so that choice is an informed one, not a guess.
