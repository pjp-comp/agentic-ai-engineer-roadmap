[← Stage 03](stage-03-tool-calling.md) · Stage 04 of 14 · **Next:** [Stage 05 →](stage-05-rag-retrieval.md)

# Stage 04 — Memory + State Management

Short-term buffers · long-term vector recall · context compression · cross-session sync · forgetting/eviction policies · memory-layer ecosystem (Letta/Zep/Cognee/Mem0)

## Why this matters

Context windows are finite and expensive; naive "append everything to the prompt" degrades both cost and accuracy past a few dozen turns. Real agents need three distinct memory types, not one — conflating them is the most common mistake here.

## Beginner focus

- Start with short-term memory first.
- Add persistent checkpointing second.
- Keep long-term memory optional in your first project.
- Skip context compression on the first pass.

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

**Runnable versions — the three tiers, built one at a time:**

| Example | Tiers it builds | What to watch |
|---|---|---|
| [`examples/stage04-memory-agent/`](../examples/stage04-memory-agent/) | Short-term + persistent | A fact stated in turn 1 ages out of the window and the agent honestly says it doesn't know — then a restart resumes the conversation intact |
| [`examples/stage04-longterm-memory-agent/`](../examples/stage04-longterm-memory-agent/) | + long-term, as a flat JSON store | A `save_fact` tool the model decides when to call — plus an honest comparison of how much better Claude judges *when* to save than a 3B local model does |
| [`examples/stage04-longterm-memory-vectorstore/`](../examples/stage04-longterm-memory-vectorstore/) | + long-term, as a real vector store | The same facts retrieved by *meaning*: "when are we kicking things off?" finds a fact saved as "start ai learning on 30th july", despite sharing no words |

Read them in that order — each one is the previous file plus one tier, so the diff between them is the lesson.

If a fourth tier is needed for very long single sessions, add a compression step (an LLM-generated running summary replacing anything that ages out of the short-term window) — but build the three above first; compression is an optimization on top, not a fourth foundational type.

## Forgetting and eviction — long-term memory needs a cleanup policy too

Long-term memory that only ever grows becomes a liability, not an asset: retrieval quality degrades as near-duplicate and stale facts crowd out the vector search, storage cost climbs unbounded, and data-retention rules (GDPR-style "right to be forgotten," or an internal policy on how long user data can be kept) require you to actually be able to delete things — not just stop referencing them. "Forgetting" here isn't a bug to avoid, it's a feature to build deliberately.

Three policies cover most real systems, and they compose (use more than one):

| Policy | What it does | When to reach for it |
|---|---|---|
| **TTL (time-to-live)** | Delete a memory after N days/months regardless of use | Compliance-driven deletion, or facts that are only relevant for a known window (e.g. "user's current project") |
| **LRU-style decay** | Lower a memory's retrieval score the longer it goes unaccessed; evict the lowest-scoring entries when storage hits a cap | Bounding storage growth without a hard deadline — lets frequently-useful facts survive indefinitely while stale ones fade |
| **Supersession on write** | When a new fact contradicts an old one (e.g. "user's email is X" replaces a prior "user's email is Y"), mark the old one superseded instead of leaving both to be retrieved | Prevents contradictory facts from both surfacing in the same retrieval and confusing the model — this one matters even in a *small* memory store |

```python
import time

class LongTermMemory:
    def __init__(self, vector_store, ttl_days: int = 180):
        self.vector_store = vector_store
        self.ttl_seconds = ttl_days * 86400

    async def remember(self, fact: str, metadata: dict, supersedes: str | None = None):
        if supersedes:
            # Mark the old fact superseded rather than deleting outright —
            # keeps an audit trail while stopping it from being retrieved.
            await self.vector_store.aupdate_metadata(supersedes, {"superseded": True})
        metadata = {**metadata, "created_at": time.time(), "superseded": False}
        await self.vector_store.aadd_texts([fact], metadatas=[metadata])

    async def recall(self, query: str, k=4) -> list[str]:
        results = await self.vector_store.asimilarity_search(
            query, k=k, filter={"superseded": False}
        )
        cutoff = time.time() - self.ttl_seconds
        return [r.page_content for r in results if r.metadata["created_at"] > cutoff]

    async def evict_expired(self):
        """Run on a schedule (e.g. nightly), not per-request — eviction is
        cleanup, not something that should add latency to a live turn."""
        cutoff = time.time() - self.ttl_seconds
        await self.vector_store.adelete(filter={"created_at": {"$lt": cutoff}})
```

A more sophisticated option some production systems use is clustering near-duplicate memories (e.g. with k-means) and replacing a cluster of redundant facts with one merged summary — this earns its complexity once you have thousands of memories with genuine overlap; for most projects, TTL + decay + supersession get you most of the benefit for a fraction of the engineering cost. Start with those three before reaching for clustering.

**What the examples cover, and what they don't.** Both long-term memory examples implement **supersession on write** — the third policy above, and the one that matters even in a small store. You can watch it work: tell [`examples/stage04-longterm-memory-vectorstore/`](../examples/stage04-longterm-memory-vectorstore/) your name is Alex, then that it's Sam, then ask — the old fact is deleted on write (a Chroma `delete`-then-`add` keyed on the fact's `key`), so recall returns one value, not two conflicting ones.

**TTL and decay are the genuine gap**, and the most worthwhile extension in this stage. The vector-store example already writes an `updated_at` timestamp on every fact, so it has the data — nothing reads it yet. Filtering recall by age, and adding an `evict_expired()` that runs on a schedule rather than per-request, is a small change against a store that's already shaped for it.

## The dedicated memory-layer ecosystem

The `LongTermMemory` class above is the pattern; you don't have to hand-roll the storage/retrieval/eviction plumbing yourself. A small set of dedicated memory-layer tools has emerged specifically for this problem — Mem0 (already cited below), Letta (the production successor to the MemGPT research project), Zep (built on a temporal knowledge graph, Graphiti), and Cognee (a self-improving knowledge-graph memory layer) are the ones worth knowing by name. Graph-based memory in particular — representing facts as a connected graph rather than a flat list, so retrieval can follow relationships ("who reports to whom," "which project depends on which") instead of pure similarity search — has moved from research-y and experimental toward genuinely production-used in real deployments over the past couple of years; that's a directional observation from watching the ecosystem, not a single citable fact, so treat it as "worth evaluating," not "the settled default." Reach for one of these instead of hand-rolling `LongTermMemory` once you need graph relationships between facts, not just a flat retrievable list — the class above is enough for most personal-agent use cases.

## Sources

| Type | Resource |
|------|----------|
| Docs | [LangChain — Memory overview](https://docs.langchain.com/oss/python/concepts/memory) |
| Tutorial | [DigitalOcean — Long-term memory with LangGraph + Mem0](https://www.digitalocean.com/community/tutorials/langgraph-mem0-integration-long-term-ai-memory) |
| Repo | [FareedKhan-dev — long-term memory reference implementation](https://github.com/FareedKhan-dev/langgraph-long-memory) |
| Tool | pgvector (Postgres), Chroma (local/embedded), Pinecone/Weaviate (managed) |
| Blog | [mem0.ai — Memory eviction and forgetting in AI agents](https://mem0.ai/blog/memory-eviction-and-forgetting-in-ai-agents) |
| Guide | [Zylos Research — Agent memory compression and state budget management](https://zylos.ai/research/2026-06-30-agent-memory-compression-state-budget-management/) |
| Tool | [Letta](https://www.letta.com/) — production successor to the MemGPT research project; agent runtime with built-in memory management |
| Tool | [Zep](https://www.getzep.com/) — memory built on Graphiti, a temporal knowledge graph, for relationship-aware recall |
| Tool | [Cognee](https://www.cognee.ai/) — self-improving, GraphRAG-style knowledge-graph memory layer |

## Done when

A 100-turn conversation stays under your token budget, the agent still recalls a fact mentioned in turn 3 (short-term), a killed-and-restarted task resumes without repeating finished work (persistent), a fact saved in one session is recalled correctly in a brand-new session days later (long-term), and a fact you deliberately mark superseded no longer shows up in retrieval results (forgetting).

---
[← Stage 03 — Tool Calling + Structured Outputs](stage-03-tool-calling.md) · [Back to roadmap](../README.md) · **Next:** [Stage 05 — RAG + Retrieval →](stage-05-rag-retrieval.md)
