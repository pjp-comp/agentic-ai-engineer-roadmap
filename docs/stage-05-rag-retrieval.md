[← Stage 04](stage-04-memory-state.md) · Stage 05 of 14 · **Next:** [Stage 06 →](stage-06-sessions-state.md)

# Stage 05 — RAG + Retrieval

Vector databases · chunking · naive vs. hybrid vs. agentic RAG · corrective RAG (CRAG) · retrieval vs. tool calls

## Why this matters

Stage 4 covered vector stores as *memory* — recalling facts an agent itself wrote down. RAG is a related but distinct problem: giving an agent access to a large body of **external** documents (news articles, filings, manuals, a knowledge base) it did not write, so it answers from real content instead of the model's training data or a hallucinated guess.

**The mistake to avoid going in:** RAG is not the answer to every "the agent needs data" problem. If the data is structured and precise — a stock price, a database row, an API response — that's a **tool call** (Stage 3), not RAG. Vector search finds text that's *semantically similar* to a query; it does not guarantee the *numerically correct* answer. Reach for RAG when the question is "what does this large body of text say," and reach for a tool when the question is "what is the current exact value of X."

## The RAG architecture spectrum

RAG isn't one design — it's a family, each layering more control (and cost) onto the same basic loop of chunk → embed → retrieve → generate:

| Type | What changes | Needs vectors? | Reach for it when |
|---|---|---|---|
| **Naive RAG** | The baseline loop: chunk docs, embed, retrieve top-k, stuff into the prompt | Yes | Getting started, FAQ-style lookups, small-to-moderate document sets |
| **Advanced/Hybrid** | Adds query rewriting, hybrid keyword+vector search (BM25 + embeddings), re-ranking | Yes | Production retrieval quality — pure semantic search misses exact terms like ticker symbols |
| **Agentic RAG** | Retrieval becomes a decision loop: search → judge "is this enough?" → retry or reroute | Yes | Ambiguous or multi-step questions, multiple data sources |
| **Corrective RAG (CRAG)** | The model **grades** retrieved chunks before trusting them; falls back if they're weak | Yes | Reducing hallucinations when the knowledge base has gaps — this is the one worth prioritizing for anything finance- or fact-sensitive |
| **Graph RAG** | Retrieves by traversing entity relationships, not just similarity | Optional | Multi-hop questions where *relationships* carry the answer |
| **Vectorless (PageIndex)** | No embeddings — the LLM reasons over a document's structure (table of contents) to pick a section | No | Long, well-structured documents (10-K filings, manuals, contracts) |

Naive RAG is the loop everyone should build first. Hybrid search and re-ranking are the two upgrades that matter most in practice — pure vector search alone regularly loses to a hybrid keyword+vector approach on anything with exact-match terms (company names, ticker symbols, dates). Corrective RAG is the one to prioritize if the cost of a wrong answer is high, since it adds an explicit "don't trust this, retry" safety net instead of confidently answering from bad context.

## Brief

Build the naive loop first, then add one upgrade at a time rather than all at once — each is independently testable.

```python
# 1. Naive RAG — the baseline loop
def naive_rag(query: str, vector_store, llm) -> str:
    chunks = vector_store.similarity_search(query, k=4)
    context = "\n\n".join(c.page_content for c in chunks)
    return llm.generate(f"Context:\n{context}\n\nQuestion: {query}")


# 2. Corrective RAG — grade before trusting, fall back if weak
def corrective_rag(query: str, vector_store, llm) -> str:
    chunks = vector_store.similarity_search(query, k=4)

    graded = [c for c in chunks if llm.grade_relevance(query, c.page_content) == "relevant"]

    if not graded:
        # Retrieved context was weak — don't answer from it. Rewrite and retry,
        # fall back to a different source, or refuse rather than guess.
        rewritten_query = llm.rewrite_query(query)
        chunks = vector_store.similarity_search(rewritten_query, k=4)
        graded = [c for c in chunks if llm.grade_relevance(query, c.page_content) == "relevant"]

    if not graded:
        return "I don't have reliable information to answer this."

    context = "\n\n".join(c.page_content for c in graded)
    return llm.generate(f"Context:\n{context}\n\nQuestion: {query}")
```

**Runnable version:** [`examples/stage05-rag-langgraph/`](../examples/stage05-rag-langgraph/) builds this exact corrective-RAG logic as a real LangGraph graph instead of nested function calls — `retrieve` → `grade` → a conditional edge that routes to `generate`, `rewrite_query` (retry once), or `refuse`, so the decision loop's *shape* is something you see in the graph definition. Runs entirely on Ollama (local chat model + local embeddings), against a small synthetic corpus so a correct answer can only come from retrieval, never from the model's training data.

**Multimodal version:** [`examples/stage05-multimodal-rag-agent/`](../examples/stage05-multimodal-rag-agent/) applies the same corrective-RAG loop to a real PDF containing text, a real table, and a real embedded chart image — using **multi-vector retrieval** (embed a generated summary of each table/image for search, but generate the final answer from the original raw content) so tables and images are genuinely searchable by meaning, not just present in the index. Also integrates Stage 6's persistent Session/Event pattern, so a research conversation over one document survives a restart. Ollama only, including a local vision model for image understanding.

## Where this fits a multi-agent build

If you're building a multi-agent system that mixes qualitative (news, filings, sentiment) and quantitative (prices, ratios, indicators) data — a stock analysis system is the canonical example — split the two cleanly: a research/sentiment agent uses RAG over news and filings; a data-fetcher and technical-analysis agent use direct tool calls (Stage 3) over a market data API. Don't embed numeric time-series data into a vector store to "make it searchable" — that's the most common RAG misuse, and it produces an agent that's confidently approximate about numbers that need to be exact.

## Sources

### Hands-on, phased (recommended starting point)

| Type | Resource |
|------|----------|
| Repo | [vector-database-learning-notes](https://github.com/pjp-comp/vector-database-learning-notes) — a phased, runnable curriculum: in-memory embeddings → real FAISS index → ANN at scale → naive RAG → hybrid search + re-ranking → agentic RAG → corrective RAG (CRAG) → vectorless RAG. Each phase has a working example; most run with no API key via a labeled fake-LLM stand-in. |
| Doc (in repo) | [`rag_architecture_types.md`](https://github.com/pjp-comp/vector-database-learning-notes/blob/main/rag_architecture_types.md) — the one-page map of every RAG type above, with a comparison table |
| Doc (in repo) | [`learning_notes.md`](https://github.com/pjp-comp/vector-database-learning-notes/blob/main/learning_notes.md) — the full plain-language guide behind every phase |
| Doc (in repo) | [`context_engineering.md`](https://github.com/pjp-comp/vector-database-learning-notes/blob/main/context_engineering.md) — how retrieval fits into assembling the LLM's full prompt |

### Concepts and production tooling

| Type | Resource |
|------|----------|
| Docs | [LangChain — Retrieval concepts](https://docs.langchain.com/oss/python/concepts/retrieval) |
| Tool | pgvector (Postgres), Chroma (local/embedded), Pinecone/Weaviate/Qdrant (managed) — same options as Stage 4's long-term memory store, since a RAG index and a memory store are the same underlying primitive used differently |
| Concept | RAGAS-style evaluation metrics for retrieval and answer quality — the retrieval-specific counterpart to Stage 10's evaluation harness |
| Reference | PageIndex — "What Is PageIndex? How to Build a Vectorless RAG System (No Embeddings, No Vector DB)" — search this title for the current write-up |

## Done when

Given a question the naive loop answers wrong or "I don't know" to because retrieval pulled irrelevant chunks, the corrective version either retries with a better query or explicitly declines — it never confidently answers from context it graded as weak.

---
[← Stage 04 — Memory + State Management](stage-04-memory-state.md) · [Back to roadmap](../README.md) · **Next:** [Stage 06 — Sessions, State + Events →](stage-06-sessions-state.md)
