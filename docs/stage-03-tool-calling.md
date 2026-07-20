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
| Spec | Model Context Protocol (MCP) spec — for dynamic/discoverable tool registries |

## Done when

A malformed tool call self-corrects within 1–2 turns instead of throwing an unhandled exception.

---
[← Stage 02 — LLM Fundamentals for Agents](stage-02-llm-fundamentals.md) · [Back to roadmap](../README.md) · **Next:** [Stage 04 — Memory + State Management →](stage-04-memory-state.md)
