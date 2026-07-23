[← Back to roadmap](../../README.md)

# Example 03 — The Same Agent, But the Tool Lives Behind MCP

This is [`examples/01-basic-agent/`](../01-basic-agent/) again — same calculator, same ReAct loop — with one deliberate change: the `calculate` tool is no longer hard-coded into the agent's Python file. It's served by a separate MCP server (`server.py`), and the agent discovers it at runtime instead of importing it.

Read this after 01, not instead of it. The point isn't a new capability — it's seeing exactly what MCP adds (and costs) over a plain tool array, using code you've already read.

## What changed vs. example 01

| | `01-basic-agent/agent.py` | `03-mcp-tool-server/` |
|---|---|---|
| Tool schema | Hand-written JSON in `TOOLS` | Auto-derived from `server.py`'s function signature via `@mcp.tool()` |
| Tool implementation | `calculate()` in the same file | `calculate()` in a separate process (`server.py`), reachable over stdio |
| How the agent finds it | Imported directly | `session.list_tools()` at startup |
| How the agent calls it | Direct Python call | `session.call_tool(name, input)` — a protocol round-trip |
| Reusable by another agent/app? | No — copy-paste the function | Yes — any MCP client can `list_tools()`/`call_tool()` against `server.py` without touching this code |

The ReAct loop itself — `messages.create()` → check `stop_reason` → run tool → append `tool_result` → repeat — is byte-for-byte the same shape as example 01. MCP changes *where the tool lives*, not *how the agent thinks*.

## What it is

- **`server.py`** — an MCP server exposing one tool (`calculate`, the identical AST-based evaluator from example 01), built with the official `mcp` Python SDK's `FastMCP` helper. `@mcp.tool()` derives the JSON schema from the function's type hints and docstring — you don't hand-write `input_schema` here, unlike in example 01.
- **`agent.py`** — the same loop as example 01, but it launches `server.py` as a subprocess over stdio, calls `list_tools()` to discover what's available, and dispatches tool calls via `call_tool()` instead of a local function reference.

## Run it

```bash
cd examples/03-mcp-tool-server
uv run agent.py "What is 23 * 47, plus 100, all divided by 3?"
```

`agent.py` loads `ANTHROPIC_API_KEY` from a `.env` file at the repo root (see [`.env.example`](../../.env.example)) via `python-dotenv` — same as [example 01](../01-basic-agent/); no `export` needed. `export ANTHROPIC_API_KEY=sk-ant-...` still works too if you prefer that.

`agent.py` starts `server.py` itself (as a subprocess, via `uv run server.py`) — you don't need to run the server separately. The first run downloads dependencies (`anthropic` + `mcp` + `python-dotenv`) into a local `.venv/` per `uv.lock`; every run after is instant.

Expected output (tool calls print to stderr so stdout stays clean):

```
  [mcp tool call] calculate({'expression': '23 * 47'})
  [mcp tool call] calculate({'expression': '1081 + 100'})
  [mcp tool call] calculate({'expression': '1181 / 3'})
393.6666666666667
```

Compare that stderr line to example 01's `[tool call] calculate(...)` — identical shape, different transport underneath.

### Run the server standalone (optional)

```bash
uv run server.py
```

It sits waiting for an MCP client on stdio (Ctrl+C to exit) — this just confirms the server itself starts cleanly, independent of any agent.

## What to look at closely

- **`_mcp_tool_to_claude_schema()`** — MCP's `Tool` object (`name`, `description`, `inputSchema`) isn't the exact shape `client.messages.create(tools=[...])` expects (`input_schema`, snake_case). This tiny adapter is the entire integration cost of using MCP tools with the Claude API directly, without the `mcp_servers`/`mcp_toolset` remote-connector pattern from [Stage 3](../../docs/stage-03-tool-calling.md#consuming-an-existing-mcp-server-the-common-case).
- **`session.call_tool(block.name, block.input)`** — this is the one line that structurally differs from example 01's `if block.name == "calculate": calculate(...)`. The agent no longer needs to know *how* to run any given tool, only that it can ask the MCP session to run it by name.
- **`is_error` still round-trips correctly** — try forcing a bad expression (e.g. ask the model something that leads it to compute `1/0`) and confirm the agent recovers via the same `is_error` pattern from [Stage 3](../../docs/stage-03-tool-calling.md), now carried through `result.isError` on the MCP response instead of being decided locally.

## When this pattern is (and isn't) worth it

Per [Stage 3's MCP section](../../docs/stage-03-tool-calling.md#mcp-vs-a-plain-api--what-mcp-actually-is): this two-process split only pays for itself if `calculate` needs to be **shared** — reused by a second agent, a teammate's app, or Claude Desktop. For a tool only this one agent will ever call, example 01's plain function is genuinely simpler and has zero protocol overhead. This example exists to make that tradeoff concrete, not to argue MCP is the default.

## Extend it (optional exercises)

1. Add a second tool to `server.py` (e.g. `word_count`) and confirm `agent.py` picks it up automatically via `list_tools()` — no changes needed on the agent side.
2. Point Claude Desktop (or any other MCP client) at `server.py` and confirm the same tool works there, unmodified — this is the reuse MCP is actually for.
3. Swap the stdio transport for `mcp.run(transport="http")` and connect over HTTP instead of a local subprocess — the setup a real shared/remote MCP server would use.
