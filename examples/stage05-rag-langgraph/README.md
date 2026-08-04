[← Back to roadmap](../../README.md)

# Stage 5 — Corrective RAG (CRAG), as a Real LangGraph Graph

[Stage 5](../../docs/stage-05-rag-retrieval.md) describes Corrective RAG as nested function calls (`naive_rag()`, then `corrective_rag()` wrapping more logic around it). This example builds the same idea as an actual LangGraph graph — retrieve, grade, generate, with a conditional edge that routes back to retry on weak retrieval — so the *shape* of the decision loop is something you can see in the graph definition, not something you have to trace through nested `if` statements.

**Ollama only** — both the chat model (`llama3.2:3b`) and the embedding model (`nomic-embed-text`) run locally. No Claude path in this example.

## How it works

The graph:

```
START -> retrieve -> grade -> (any chunk relevant?)
                                    |-- yes --------------> generate -> END
                                    |-- no, haven't retried -> rewrite_query -> retrieve (loop back)
                                    |-- no, already retried -> refuse -> END
```

- **`retrieve`** — embeds the query, runs `similarity_search(k=3)` against a small local ChromaDB vector store (in-memory, rebuilt on every run — no persistent index to manage for this demo).
- **`grade`** — the *corrective* part of CRAG. Each retrieved chunk is graded against the **original** question by asking the model a yes/no relevance question — not trusting retrieval blindly just because it returned k results.
- **`rewrite_query`** (conditional) — if nothing graded relevant and this is the first attempt, ask the model to rewrite the query and retry retrieval once. A vocabulary mismatch between the question and the corpus is a common, fixable failure that a straight retry (same query) wouldn't help with.
- **`generate`** — build the final answer from only the graded-relevant chunks, with source citations.
- **`refuse`** (conditional) — if nothing is relevant even after one retry, say so plainly instead of guessing. No LLM call here — there's nothing worth reasoning over.

## Why a synthetic corpus

`agent.py` embeds four short, invented documents about a fictional company ("Northwind Traders") directly in the script — deliberately synthetic and specific (invented numbers, invented executives) so a correct answer can **only** come from retrieval. There's no way for the model to "already know" this from training data, which makes it unambiguous when the graph is actually using retrieved context versus pattern-matching from memory.

## Run it

```bash
cd examples/stage05-rag-langgraph
ollama pull llama3.2:3b        # one-time, ~2GB, shared with other examples
ollama pull nomic-embed-text   # one-time, ~274MB, shared with stage04-longterm-memory-vectorstore
uv run agent.py "What was Northwind Traders' revenue in fiscal 2025?"
```

Expected output — retrieval finds the right document, grading filters out the two irrelevant ones, generation cites its source:

```
  [retrieve] query="What was Northwind Traders' revenue in fiscal 2025?" -> 3 chunk(s)
  [grade] annual_report_fy2025.txt: relevant
  [grade] company_overview.txt: not relevant
  [grade] q3_earnings_call.txt: not relevant
$184.3 million [annual_report_fy2025.txt]
```

Force the refusal path — nothing in the corpus is relevant, so it should retry once, still fail, and refuse rather than hallucinate:

```bash
uv run agent.py "What is the capital of France?"
```

Try a question that needs the grading step to actually discriminate (both retrieved chunks mention the same name, only one is relevant to *this* question):

```bash
uv run agent.py "Who is the CFO of Northwind Traders and what did she attribute the margin improvement to?"
```

## What to look at closely

- **`route_after_grading()` is the entire corrective-RAG decision**, expressed as a plain function returning a string that LangGraph's `add_conditional_edges` uses to pick the next node — not an `if/else` buried inside a bigger function. This is the "graph" layer from the [harness/loop/graph framework](../../docs/stage-07-single-agent.md#harness-loop-and-graph--the-current-framing-for-how-agents-get-built): the control flow is data (a routing table), not code you have to read top-to-bottom to understand.
- **Grading happens against `original_query`, never the rewritten one** — if grading used the rewritten query, a bad rewrite could make irrelevant chunks look relevant to *that* wording while still not answering what the user actually asked. Grading against the original keeps the bar honest regardless of what the query became.
- **`MAX_RETRIES = 1`** — same "graceful degradation" principle as [Stage 7](../../docs/stage-07-single-agent.md)'s `MAX_ITERATIONS`: an agent that can retry forever on a query that will never match anything is a liability, not a feature. One rewrite-and-retry catches the common "wrong vocabulary" case; it's not meant to catch "the corpus genuinely doesn't have this."
- **Two separate models, two separate jobs** — `LOCAL_MODEL` (chat) does generation and grading; `EMBEDDING_MODEL` (`nomic-embed-text`) only turns text into vectors. Conflating "the model that answers" with "the model that embeds" is a common beginner mistake Stage 5 warns about — they're not the same job, and most real systems use different models for each.
- **The vector store is rebuilt from scratch on every run** — fine for a 4-document demo; a real system would build the index once and reuse it. See [Stage 4's long-term-memory vector store example](../stage04-longterm-memory-vectorstore/) for a persisted-index version of the same ChromaDB pattern.

## Where this goes next

This is "agentic RAG" from [Stage 5's spectrum table](../../docs/stage-05-rag-retrieval.md#the-rag-architecture-spectrum) — retrieval as a decision loop, not a single fixed pass. The table's other upgrades (hybrid keyword+vector search, re-ranking, Graph RAG) would each add a node or two to this same graph shape, not require a different architecture. If you want to see the same "grade before trusting" instinct applied to *saved facts* instead of an external corpus, see [Stage 4's forgetting/supersession policies](../../docs/stage-04-memory-state.md#forgetting-and-eviction--long-term-memory-needs-a-cleanup-policy-too) — a related but distinct problem (bad retrieval vs. stale memory).
