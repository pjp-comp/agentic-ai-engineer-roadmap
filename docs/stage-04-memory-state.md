[← Stage 03](stage-03-tool-calling.md) · Stage 04 of 14 · **Next:** [Stage 05 →](stage-05-rag-retrieval.md)

# Stage 04 — Memory + State Management

Short-term buffers · long-term vector recall · context compression · cross-session sync

## Why this matters

Context windows are finite and expensive; naive "append everything to the prompt" degrades both cost and accuracy past a few dozen turns. Real agents need three distinct memory types, not one — conflating them is the most common mistake here.

## The three memory types

| Type | Lifespan | Where it lives | Purpose |
|------|----------|-----------------|---------|
| **Short-term** | One conversation/session | In-process (a `deque`, a list) | The last N raw turns — verbatim, no compression, so recent context stays exact |
| **Persistent (session state)** | One task, possibly across process restarts | A checkpoint store (Redis, Postgres, disk) | Lets a long-running or interrupted task resume without re-doing work — this is what Stage 5's checkpointing depends on |
| **Long-term (cross-session)** | Forever, across unrelated sessions | A vector store (pgvector, Chroma, Pinecone) | Facts, preferences, and learnings that should survive into a conversation that hasn't happened yet |

These solve different problems. Short-term memory keeps the current exchange coherent. Persistent memory keeps a single long task alive across restarts. Long-term memory is what makes an agent remember *you* the next time you show up. A system that only implements one of the three will either forget everything overnight (no long-term) or blow its context budget on turn 50 (no short-term compression).

## Brief

Implement all three as separate, composable pieces — don't merge them into one class, since they have different lifetimes and failure modes.

```python
from collections import deque

# 1. Short-term: rolling window, exact recall of recent turns
class ShortTermMemory:
    def __init__(self, window=10):
        self.buffer = deque(maxlen=window)

    def add(self, role: str, content: str):
        self.buffer.append({"role": role, "content": content})

    def get(self):
        return list(self.buffer)


# 2. Persistent: task-scoped state that survives a process restart
class PersistentState:
    def __init__(self, task_id: str, store):
        self.task_id = task_id
        self.store = store  # e.g. Redis client, or a Postgres row

    def checkpoint(self, state: dict):
        self.store.set(f"task:{self.task_id}", state)

    def resume(self) -> dict | None:
        return self.store.get(f"task:{self.task_id}")


# 3. Long-term: durable facts, retrieved by semantic similarity
class LongTermMemory:
    def __init__(self, vector_store):
        self.vector_store = vector_store  # e.g. Chroma, pgvector

    async def remember(self, fact: str, metadata: dict):
        await self.vector_store.aadd_texts([fact], metadatas=[metadata])

    async def recall(self, query: str, k=4) -> list[str]:
        results = await self.vector_store.asimilarity_search(query, k=k)
        return [r.page_content for r in results]


# Composed: what the agent actually calls each turn
class AgentMemory:
    def __init__(self, task_id: str, checkpoint_store, vector_store):
        self.short_term = ShortTermMemory()
        self.persistent = PersistentState(task_id, checkpoint_store)
        self.long_term = LongTermMemory(vector_store)

    async def get_context(self, query: str) -> list[str]:
        recalled = await self.long_term.recall(query)
        return [*recalled, *self.short_term.get()]
```

If a fourth tier is needed for very long single sessions, add a compression step (an LLM-generated running summary replacing anything that ages out of the short-term window) — but build the three above first; compression is an optimization on top, not a fourth foundational type.

## Sources

| Type | Resource |
|------|----------|
| Docs | [LangChain — Memory overview](https://docs.langchain.com/oss/python/concepts/memory) |
| Tutorial | [DigitalOcean — Long-term memory with LangGraph + Mem0](https://www.digitalocean.com/community/tutorials/langgraph-mem0-integration-long-term-ai-memory) |
| Repo | [FareedKhan-dev — long-term memory reference implementation](https://github.com/FareedKhan-dev/langgraph-long-memory) |
| Tool | pgvector (Postgres), Chroma (local/embedded), Pinecone/Weaviate (managed) |

## Done when

A 100-turn conversation stays under your token budget, the agent still recalls a fact mentioned in turn 3 (short-term), a killed-and-restarted task resumes without repeating finished work (persistent), and a fact saved in one session is recalled correctly in a brand-new session days later (long-term).

---
[← Stage 03 — Tool Calling + Structured Outputs](stage-03-tool-calling.md) · [Back to roadmap](../README.md) · **Next:** [Stage 05 — RAG + Retrieval →](stage-05-rag-retrieval.md)
