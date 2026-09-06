[← Stage 02](stage-02-llm-fundamentals.md) · Stage 03 of 14 · **Next:** [Stage 04 →](stage-04-memory-state.md)

# Stage 03 — Tool Calling + Structured Outputs

Pydantic validation · function calling schemas · error recovery · dynamic tool discovery · MCP vs. plain APIs · Skills and progressive disclosure · computer-use tools

## Why this matters

Tool calling is the interface between model reasoning and the real world. Loose schemas and silent validation failures are the single biggest source of "the agent did something weird" bugs — Pydantic makes the contract explicit and catches malformed calls before they execute.

## Beginner focus

- Build only one tool first.
- Validate one input schema with Pydantic.
- Handle one validation error path.
- Skip MCP until the basic tool loop works.

## Brief

Define a tool with a strict Pydantic input schema, force the model to call it via `tool_choice`, validate the arguments, and on a validation error, feed the error back to the model as a correction turn instead of crashing.

The mechanics below (`tool_call.input`, `tool_choice`) are Claude's specific field names — every tool-calling API has an equivalent shape (OpenAI nests arguments as a JSON string under `tool_calls[i].function.arguments`; Gemini uses `functionCall.args`), but the validate-then-feed-the-error-back pattern is the same regardless of provider:

**Claude API example:**

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
| Docs | [OpenAI — Function calling](https://developers.openai.com/api/docs/guides/function-calling) — the same concept (schema-defined tools the model can invoke) under OpenAI's naming and JSON-schema conventions |
| Docs | [Claude Platform Docs — Structured outputs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs) |
| Docs | [Pydantic v2 — official docs](https://docs.pydantic.dev/latest/) |
| Library | [Instructor — structured extraction on top of Claude](https://python.useinstructor.com/integrations/anthropic/) |
| Spec | [Model Context Protocol — official spec](https://modelcontextprotocol.io/) — see the MCP vs. API section below |
| Course | [DeepLearning.AI × Anthropic — Agent Skills with Anthropic](https://www.deeplearning.ai/courses/agent-skills-with-anthropic) — reusable Skills as an alternative to embedding workflow logic directly in prompts |

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

Most developers *use* MCP servers before they ever build one. MCP itself is vendor-neutral — the protocol works the same regardless of which model client is speaking it — but how a given client connects to a remote MCP server is provider-specific API surface. Claude's Messages API can connect directly, no client-side tool execution loop needed:

**Claude API example:**

```python
response = client.beta.messages.create(
    model="claude-opus-5",
    max_tokens=1024,
    betas=["mcp-client-2025-11-20"],
    mcp_servers=[
        {"type": "url", "url": "https://api.githubcopilot.com/mcp/", "name": "github"}
    ],
    tools=[{"type": "mcp_toolset", "mcp_server_name": "github"}],
    messages=[{"role": "user", "content": "List open issues in this repo"}],
)
```

Two things that trip people up here: the `mcp_servers` entry and the `mcp_toolset` entry are **both** required — passing only `mcp_servers` is rejected as a validation error — and `betas=[...]` carries a dated beta header that rotates as the feature matures. Check the current value in the [MCP connector docs](https://platform.claude.com/docs/en/agents-and-tools/mcp-connector) rather than copying the string above verbatim; a stale beta header is the most common reason this snippet stops working months later.

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

**Runnable version:** [`examples/stage03-mcp-tool-server/`](../examples/stage03-mcp-tool-server/) takes this exact sketch further — it's example 01's calculator agent with the tool moved behind a real MCP server, so you can run both the server and a Claude-driven client and see the discovery + `call_tool()` round-trip actually happen.

## Skills — packaging procedural knowledge instead of stuffing the prompt

Tools give the model new *capabilities*. There's a separate problem tools don't solve: giving the model **procedural knowledge** — how your company formats a report, the six steps of your deploy checklist, the conventions of your codebase. The instinctive fix is to put all of it in the system prompt, which fails for a reason [Stage 2's context-confusion failure mode](stage-02-llm-fundamentals.md#the-four-context-failure-modes--a-debugging-vocabulary) names precisely: every instruction you add is present on *every* call, including the 95% of calls where it's irrelevant, and irrelevant context degrades the model's choices.

**Skills** solve this with **progressive disclosure**: procedural knowledge lives in files, and the model loads only what the current task needs. The mechanism is three-tier:

| Tier | What's always in context | When it loads |
|---|---|---|
| **1. Metadata** | Just a name and one-line description of each skill | Always — this is the only permanent cost, a few dozen tokens per skill |
| **2. Body** | The skill's actual instructions | Only when the model judges the description relevant to the current task |
| **3. Linked resources** | Reference files, scripts, templates the skill points to | Only when the loaded skill actually needs them |

The payoff: you can have fifty skills available while paying the context cost of fifty one-line descriptions, not fifty full procedures. A skill is just a folder with a markdown file and optional supporting files — no framework, no API, no fine-tuning:

```
skills/
  quarterly-report/
    SKILL.md          # frontmatter: name + description; body: the procedure
    template.xlsx     # loaded only if the procedure references it
```

**How this differs from the neighbours it gets confused with:**

- **vs. a tool** — a tool is something the model *calls* and gets a result back from; a skill is something the model *reads* to know how to proceed. A skill can tell the model which tools to call, in what order.
- **vs. RAG** (Stage 5) — RAG retrieves *facts* to answer a question. Skills load *procedures* to perform a task. Similar retrieval mechanics, different content and different trigger.
- **vs. fine-tuning** — fine-tuning bakes behavior into weights: expensive, slow to iterate, opaque when wrong. A skill is a text file you edit and re-run. For teaching a model *how your organization does something*, skills are almost always the right tool now; fine-tuning is for changing the model's fundamental capabilities, not its procedures.
- **vs. MCP** — orthogonal, and they compose. MCP delivers tools; skills deliver the know-how for using them well. A skill can reference MCP-provided tools by name.

**When to reach for it:** you have a procedure that's long, that applies to only a fraction of requests, and that you expect to revise. If it's short and always relevant, it belongs in the system prompt — a skill would just be indirection. The general principle is the one worth taking away even if you never write a `SKILL.md`: **context should be loaded on demand, not held permanently**, and that principle applies to tool schemas and retrieved documents just as much as to instructions.

## A third paradigm: computer-use tools

Plain API calls and MCP both assume the tool has an API — a defined schema the model fills in. Some tasks genuinely don't have one: a legacy internal app with no API, a site that only exposes a UI. **Computer-use tools** are the answer for that case: instead of a schema, the model perceives a screenshot and issues GUI actions (click, type, scroll) — same `tool_use`/`tool_result` mechanics as everything else in this stage, just with pixel coordinates and screen state instead of typed parameters. It's a last resort, not a default — reach for it only when there's genuinely no API to call, since it's slower and less reliable than a schema-defined tool. See [Stage 7's note on computer-use and voice agents](stage-07-single-agent.md#beyond-requestresponse-computer-use-and-voice-agents) for where this fits alongside the request/response tool-calling this stage teaches.

---
[← Stage 02 — LLM Fundamentals for Agents](stage-02-llm-fundamentals.md) · [Back to roadmap](../README.md) · **Next:** [Stage 04 — Memory + State Management →](stage-04-memory-state.md)
