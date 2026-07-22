[← Stage 02](stage-02-llm-fundamentals.md) · Stage 03 of 12 · **Next:** [Stage 04 →](stage-04-memory-state.md)

# Stage 03 — Tool Calling + Structured Outputs

Pydantic validation · function calling schemas · error recovery · dynamic tool discovery

## Why this matters

Tool calling is the interface between model reasoning and the real world. Loose schemas and silent validation failures are the single biggest source of "the agent did something weird" bugs — Pydantic makes the contract explicit and catches malformed calls before they execute.

## Brief

Define a tool with a strict Pydantic input schema, force the model to call it via `tool_choice`, validate the arguments, and on a validation error, feed the error back to the model as a correction turn instead of crashing.

```python
class SearchArgs(BaseModel):
    query: str = Field(min_length=1, max_length=200)
    max_results: int = Field(default=5, ge=1, le=20)

try:
    args = SearchArgs.model_validate(tool_call.input)
except ValidationError as e:
    # feed e back to the model as a tool_result error
    # so it can retry with corrected arguments
    return {"role": "tool", "content": str(e), "is_error": True}
```

## Sources

| Type | Resource |
|------|----------|
| Docs | [Claude Platform Docs — Tool use overview](https://platform.claude.com/docs/en/build-with-claude/tool-use) |
| Docs | [Claude Platform Docs — Structured outputs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs) |
| Docs | [Pydantic v2 — official docs](https://docs.pydantic.dev/latest/) |
| Library | [Instructor — structured extraction on top of Claude](https://python.useinstructor.com/integrations/anthropic/) |
| Spec | [Model Context Protocol — official spec](https://modelcontextprotocol.io/) — see the MCP vs. API section below |

## Done when

A malformed tool call self-corrects within 1–2 turns instead of throwing an unhandled exception.

## MCP vs. a plain API — what MCP actually is

MCP (Model Context Protocol) is not a replacement for REST APIs, and it isn't magic — it's a **standard interface between an agent and a tool server**, so the same tool works across any MCP-compatible client (Claude, an IDE, another agent framework) without custom glue code for each one. The comparison that actually matters:

| | **Calling an API directly** (what Stage 3's brief does) | **MCP** |
|---|---|---|
| **Who defines the tool schema** | You — hand-write the `input_schema` JSON for every tool, per client | The MCP server exposes its own schema; any MCP client discovers it automatically |
| **Reuse across agents/apps** | None — every app that wants the tool re-implements the wrapper | One server, any number of clients (Claude Desktop, your agent, a teammate's agent) |
| **Auth model** | Whatever the API requires — you handle it yourself | Standardized in the protocol (OAuth, static tokens); vault-backed in Managed Agents |
| **Discovery** | Static — you hard-code which tools exist | Dynamic — a client can query `list_tools` at runtime |
| **When to reach for it** | A single tool, used by one app, that doesn't need to be shared | A tool (or set of tools) meant to be reused across multiple agents/clients, or a third-party integration (GitHub, Slack, Linear) that already ships an MCP server |

**In practice: don't build an MCP server for a tool only your one agent will ever call.** That's what the plain `tools=[{...}]` array from the brief above is for — it's less code and there's no protocol overhead. Reach for MCP when the tool needs to be *shared* — reused by more than one agent or app, or when you're consuming a third-party service that already publishes an MCP server (so you get the integration for free instead of hand-rolling API calls).

### Consuming an existing MCP server (the common case)

Most developers *use* MCP servers before they ever build one. Claude's Messages API can connect to a remote MCP server directly — no client-side tool execution loop needed:

```python
response = client.beta.messages.create(
    model="claude-opus-4-8",
    max_tokens=1024,
    betas=["mcp-client-2025-11-20"],
    mcp_servers=[
        {"type": "url", "url": "https://api.githubcopilot.com/mcp/", "name": "github"}
    ],
    tools=[{"type": "mcp_toolset", "mcp_server_name": "github"}],
    messages=[{"role": "user", "content": "List open issues in this repo"}],
)
```

### Building a minimal MCP server

If you do have a tool worth sharing, an MCP server is a small program that speaks the protocol over stdio or HTTP and exposes a `list_tools` / `call_tool` interface. Sketch, using the official Python SDK:

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("weather-server")

@mcp.tool()
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"72°F and sunny in {city}"  # replace with a real API call

if __name__ == "__main__":
    mcp.run(transport="stdio")
```

That's the whole server — `mcp.tool()` auto-derives the schema from the function signature (same idea as Stage 3's Pydantic validation, but the schema is now discoverable by any MCP client, not just your own code).

---
[← Stage 02 — LLM Fundamentals for Agents](stage-02-llm-fundamentals.md) · [Back to roadmap](../README.md) · **Next:** [Stage 04 — Memory + State Management →](stage-04-memory-state.md)
