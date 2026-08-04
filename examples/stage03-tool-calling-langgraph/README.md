[← Back to roadmap](../../README.md)

# Example 01b — The Same Agent, Rebuilt on LangGraph

This is [`examples/stage03-tool-calling/`](../stage03-tool-calling/) again — same tool, same task, same iteration cap — rebuilt on [LangGraph](https://docs.langchain.com/oss/python/langgraph/overview) instead of a hand-written `while` loop.

Read `stage03-tool-calling/` first. This example exists to answer one question directly: **what does a framework actually replace?** Every line of the raw loop maps to something LangGraph does for you here — read them side by side and the framework stops being a black box.

**Runs against a free local Ollama model by default** (`USE_LOCAL_MODEL=true`) — same switch as every other example, set `USE_LOCAL_MODEL=false` for Claude instead.

## What changed vs. example 01

| Example 01 (raw API) | This example (LangGraph) | What it replaces |
|---|---|---|
| `messages = [...]` list you append to by hand | `MessagesState` | The conversation-history list and its append calls |
| Hand-written `TOOLS` dict with `input_schema` JSON | `@tool` decorator on `calculate()` | Manually writing the JSON Schema — LangGraph derives it from the function signature + docstring |
| `if block.type == "tool_use": ... calculate(...)` dispatch loop | `ToolNode(TOOLS)` | The manual "find every tool_use block, run it, build tool_result blocks" logic |
| `if response.stop_reason != "tool_use": return ...` | `tools_condition` (a conditional graph edge) | The branch that decided whether to keep looping or return the final answer |
| `for step in range(MAX_ITERATIONS): ...` | `StateGraph` with `agent` → `tools` → `agent` edges + `recursion_limit` | The loop itself — iteration is expressed as graph edges, not a `for` statement |

The tool (`calculate`, the same AST-based evaluator — no `eval()`) and the model (`claude-haiku-4-5`) are identical. Nothing about *what* the agent can do changed; only *how the loop is expressed* changed.

## What it is

- **One tool**: `calculate`, decorated with `@tool` — LangGraph reads the type hint (`expression: str`) and docstring to build the schema automatically, instead of the hand-written JSON in example 01.
- **A two-node graph**: `agent` (calls Claude via `ChatAnthropic`) and `tools` (runs whichever tool Claude asked for), wired with a conditional edge (`tools_condition`) that loops back to `agent` after every tool call and exits to `END` once Claude answers in plain text.
- **~90 lines total**, most of it the same tool implementation and comments as example 01 — the graph wiring itself is about 10 lines.

## Run it

```bash
cd examples/stage03-tool-calling-langgraph
uv run agent.py "What is 23 * 47, plus 100, all divided by 3?"
```

Uses the same repo-root `.env` as [example 01](../stage03-tool-calling/) (see [`.env.example`](../../.env.example)) via `python-dotenv` — no `export` needed. `export ANTHROPIC_API_KEY=sk-ant-...` still works too.

Expected output (tool calls print to stderr so stdout stays clean):

```
  [tool call] calculate({'expression': '(23 * 47 + 100) / 3'})
**(23 × 47 + 100) ÷ 3 = 393.67** (approximately)

To break it down:
- 23 × 47 = 1081
- 1081 + 100 = 1181
- 1181 ÷ 3 ≈ 393.67
```

Try it without a math question too (`uv run agent.py "What's the capital of France?"`) — `tools_condition` routes straight to `END` with zero tool calls, same as `stop_reason` never becoming `"tool_use"` in example 01.

## What to look at closely

- **`llm.bind_tools(TOOLS)`** — this is where the `@tool`-decorated function's auto-derived schema gets attached to the model, replacing the `tools=TOOLS` kwarg passed on every `client.messages.create()` call in example 01.
- **`call_model()` prepends the system prompt to `state["messages"]` on every call** — LangGraph's `MessagesState` doesn't have a separate system-prompt slot the way `client.messages.create(system=...)` does; you inject it into the message list yourself. Small but real API-shape difference to notice.
- **`graph.add_conditional_edges("agent", tools_condition)`** — this one call *is* the branch from example 01's `if response.stop_reason != "tool_use": return ...`. It's worth opening LangGraph's source for `tools_condition` once to see it's genuinely just checking `messages[-1].tool_calls`, nothing more exotic.
- **`recursion_limit` vs. `MAX_ITERATIONS`** — a full "agent turn, then tool turn" cycle is two graph steps (`agent` → `tools`), so the recursion limit is set to `MAX_ITERATIONS * 2` to give the same effective iteration budget as example 01's for-loop.

## Where this goes next

This is the same building block as example 01, in the shape you'll actually use once you leave raw API calls behind — [Stage 8 (Multi-Agent Orchestration)](../../docs/stage-08-multi-agent.md) builds directly on `StateGraph`, so a two-agent supervisor graph is a small extension of the two-node graph here, not a new mental model.

## Extend it (optional exercises)

1. Add a second `@tool`-decorated function and confirm `ToolNode` picks it up automatically from the `TOOLS` list — no dispatch code to write, unlike example 01's `if block.name == "calculate": ... else: ...`.
2. Print `app.get_graph().draw_mermaid()` and look at the diagram — a 5-line graph description renders into a visual you can hand to someone who's never seen the code.
3. Swap `MAX_ITERATIONS * 2` for a much smaller number (e.g. `2`) and ask a question that needs multiple tool calls — watch it hit `GraphRecursionError` instead of example 01's graceful "(gave up: exceeded MAX_ITERATIONS...)" string, and consider which failure mode you'd rather ship.
