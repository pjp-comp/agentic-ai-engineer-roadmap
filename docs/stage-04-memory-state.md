[← Stage 03](stage-03-tool-calling.md) · Stage 04 of 12 · **Next:** [Stage 05 →](stage-05-single-agent.md)

# Stage 04 — Memory + State Management

Short-term buffers · long-term vector recall · context compression · cross-session sync

## Why this matters

Context windows are finite and expensive; naive "append everything to the prompt" degrades both cost and accuracy past a few dozen turns. Real agents need a tiered memory: hot buffer for the current task, compressed summaries for older turns, and vector recall for facts that must survive across sessions.

## Brief

Implement a three-tier memory manager: a rolling window of the last N raw messages, an LLM-generated running summary that replaces anything older, and a pgvector/Chroma store for durable facts retrieved by semantic similarity at the start of each turn.

```python
class TieredMemory:
    def __init__(self, window=10):
        self.buffer = deque(maxlen=window)
        self.summary = ""
        self.vector_store = Chroma(...)

    async def get_context(self, query: str):
        recalled = await self.vector_store.asimilarity_search(query, k=4)
        return [self.summary, *recalled, *self.buffer]
```

## Sources

| Type | Resource |
|------|----------|
| Docs | [LangChain — Memory overview](https://docs.langchain.com/oss/python/concepts/memory) |
| Tutorial | [DigitalOcean — Long-term memory with LangGraph + Mem0](https://www.digitalocean.com/community/tutorials/langgraph-mem0-integration-long-term-ai-memory) |
| Repo | [FareedKhan-dev — long-term memory reference implementation](https://github.com/FareedKhan-dev/langgraph-long-memory) |
| Tool | pgvector (Postgres), Chroma (local/embedded), Pinecone/Weaviate (managed) |

## Done when

A 100-turn conversation stays under your token budget and the agent still recalls a fact mentioned in turn 3.

---
[← Stage 03 — Tool Calling + Structured Outputs](stage-03-tool-calling.md) · [Back to roadmap](../README.md) · **Next:** [Stage 05 — Single-Agent Workflows →](stage-05-single-agent.md)
