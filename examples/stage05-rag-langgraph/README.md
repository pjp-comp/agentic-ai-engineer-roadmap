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

- **`retrieve`** — embeds the query, runs `similarity_search(k=3)` against a small local ChromaDB vector store, persisted to disk in this example's own folder (`.chroma_crag/`) — built once by `build_index.py`, reused on every `agent.py` run after that.
- **`grade`** — the *corrective* part of CRAG. Each retrieved chunk is graded against the **original** question by asking the model a yes/no relevance question — not trusting retrieval blindly just because it returned k results.
- **`rewrite_query`** (conditional) — if nothing graded relevant and this is the first attempt, ask the model to rewrite the query and retry retrieval once. A vocabulary mismatch between the question and the corpus is a common, fixable failure that a straight retry (same query) wouldn't help with.
- **`generate`** — build the final answer from only the graded-relevant chunks, with source citations.
- **`refuse`** (conditional) — if nothing is relevant even after one retry, say so plainly instead of guessing. No LLM call here — there's nothing worth reasoning over.

## How a vector database actually works

This is the part that looks like magic from the outside — three real steps, none of them magic:

**1. Embedding — text becomes a point in space.** `nomic-embed-text` (the embedding model, run locally via Ollama) reads a document's text and outputs a **vector**: a fixed-length list of 768 floating-point numbers. That vector is a *coordinate* — a single point in a 768-dimensional space. The model was trained so that documents with **similar meaning** land at **nearby points**, even when they share no words in common. `"the CEO resigned"` and `"the chief executive stepped down"` end up close together in that space; `"the CEO resigned"` and `"it rained all afternoon"` end up far apart. This is fundamentally different from a keyword index (like a database `LIKE` query or full-text search) — it's comparing *meaning*, not *substring overlap*.

**2. Storing — the index.** Chroma (the vector database this example uses) stores each vector next to the document's original text and metadata (here, just `{"source": "monkey_and_crocodile_part1.txt"}`), in a structure built for one specific operation: given a new vector, quickly find the *k* stored vectors closest to it. At 5 documents this would be instant even compared one-by-one; the actual point of a real vector database's index structure is staying fast at thousands or millions of vectors, where brute-force comparison would be too slow.

**3. Retrieving — nearest neighbors.** A question gets embedded with the **same** model into the **same** space, then Chroma finds the stored vectors nearest to it — "nearest" measured by cosine similarity (the angle between two vectors), which is why two pieces of text that mean the same thing stay close even if their vectors have different magnitudes. `retrieve()`'s `similarity_search(query, k=3)` call is doing exactly this: embed the query, find the 3 nearest documents. **Nearest is not the same as relevant** — vector search returns candidates ranked by similarity, not a verified answer, which is exactly the gap `grade()` exists to close: a chunk can be the "closest" match in vector space and still not actually answer the question, and CRAG's whole premise is not trusting proximity alone.

Two separate pieces of software are doing two separate jobs here, and mixing them up is a common beginner mistake: `OllamaEmbeddings` (below) only converts text → vector; `Chroma` only stores vectors and answers nearest-neighbor queries — it doesn't know how to embed anything itself, which is why it's constructed with an `embedding_function` passed in. Swapping either independently (a different embedding model, or a different vector database like pgvector or Pinecone) is a real, ordinary thing to do precisely because they're decoupled this way.

## Why this corpus

`agent.py` embeds five short passages from the Panchatantra — the ancient Indian collection of animal fables — directly in the script: the two-part story of "The Monkey and the Crocodile," its stated moral, background on the Panchatantra collection itself, and a second, unrelated tale ("The Tortoise and the Geese") that shares the same *kind* of content (an animal fable) without answering the same questions. That overlap is deliberate — a corpus where everything is either "obviously relevant" or "obviously unrelated" doesn't actually exercise the grading step. Two documents about *different* monkey-and-crocodile-adjacent content (the story itself vs. its moral) forces `grade` to discriminate on what the question actually asks, not just which document mentions the same character names.

## Run it

```bash
cd examples/stage05-rag-langgraph
ollama pull llama3.2:3b        # one-time, ~2GB, shared with other examples
ollama pull nomic-embed-text   # one-time, ~274MB, shared with stage04-longterm-memory-vectorstore
uv run build_index.py          # build the index ONCE, explicitly
```

```
Embedding 5 document(s) with nomic-embed-text via Ollama...
Done. Index written to .chroma_crag/ — 5 document(s), ready for agent.py to query without re-embedding.
```

Now query it, as many times as you want — no re-embedding happens on any of these:

```bash
uv run agent.py "Why did the monkey refuse to give the crocodile his heart?"
```

Expected output — retrieval finds the right passage, grading filters out the other two, generation cites its source:

```
  [retrieve] query="Why did the monkey refuse to give the crocodile his heart?" -> 3 chunk(s)
  [grade] monkey_and_crocodile_part2.txt: relevant
  [grade] monkey_and_crocodile_part1.txt: not relevant
  [grade] monkey_and_crocodile_moral.txt: not relevant
The monkey refused to give the crocodile his heart because he had realized that the crocodile's plan was to kill him for his heart, as admitted by the crocodile himself while carrying the monkey on his back. [monkey_and_crocodile_part2.txt]
```

Force the refusal path — nothing in the corpus is relevant, so it should retry once, still fail, and refuse rather than hallucinate:

```bash
uv run agent.py "What is the capital of France?"
```

Try a question that needs the grading step to actually discriminate between two different stories in the same corpus:

```bash
uv run agent.py "Why did the tortoise fall from the sky?"
```

This correctly pulls `tortoise_and_geese.txt` and rejects both monkey-and-crocodile chunks, even though all three are Panchatantra animal fables about a character being carried by another animal.

If you edit `_DOCS` (add, remove, or change a passage), the persisted index won't know — nothing here diffs old vs. new documents automatically. Force a fresh embed:

```bash
uv run build_index.py --force
```

(`uv run agent.py --rebuild "some question"` does the same wipe-and-reembed inline, as a shortcut, if you'd rather not run a separate command.)

What happens if you skip `build_index.py` entirely and just run `agent.py` straight away — it still works, but warns you it's taking a slower path:

```bash
rm -rf .chroma_crag && uv run agent.py "some question"
```

```
  [vector store] WARNING: no index found at .chroma_crag/ -- embedding 5 document(s) now as a fallback. Run `uv run build_index.py` first next time to skip this.
```

## What to look at closely

- **`route_after_grading()` is the entire corrective-RAG decision**, expressed as a plain function returning a string that LangGraph's `add_conditional_edges` uses to pick the next node — not an `if/else` buried inside a bigger function. This is the "graph" layer from the [harness/loop/graph framework](../../docs/stage-07-single-agent.md#harness-loop-and-graph--the-current-framing-for-how-agents-get-built): the control flow is data (a routing table), not code you have to read top-to-bottom to understand.
- **Grading happens against `original_query`, never the rewritten one** — if grading used the rewritten query, a bad rewrite could make irrelevant chunks look relevant to *that* wording while still not answering what the user actually asked. Grading against the original keeps the bar honest regardless of what the query became.
- **A real small-model grading quirk this surfaced**: the first version of `grade`'s prompt asked "does this text contain information that helps answer the question?" — a bar `llama3.2:3b` graded too strictly, rejecting a passage that plainly described the answer's events just because it didn't state the answer as a single explicit sentence. The fix was rephrasing to "is this text relevant to the question's topic, characters, or events, even if it doesn't state the answer outright" — a genuinely easier and more honest judgment for a 3B model to make correctly, and arguably the more correct division of labor anyway: `grade` should filter for *topical relevance*, and `generate` (a second, separate call) is what actually reasons over the content to produce an answer.
- **`MAX_RETRIES = 1`** — same "graceful degradation" principle as [Stage 7](../../docs/stage-07-single-agent.md)'s `MAX_ITERATIONS`: an agent that can retry forever on a query that will never match anything is a liability, not a feature. One rewrite-and-retry catches the common "wrong vocabulary" case; it's not meant to catch "the corpus genuinely doesn't have this."
- **Two separate models, two separate jobs** — `LOCAL_MODEL` (chat) does generation and grading; `EMBEDDING_MODEL` (`nomic-embed-text`) only turns text into vectors. Conflating "the model that answers" with "the model that embeds" is a common beginner mistake Stage 5 warns about — they're not the same job, and most real systems use different models for each.
- **Indexing and querying are two separate scripts, on purpose** — `build_index.py` embeds `_DOCS` once and exits; `agent.py` only ever *opens* what's already in `.chroma_crag/`. This mirrors a real production RAG system, where an offline indexing job and an online query service are genuinely different processes with different responsibilities (and often different schedules — reindexing nightly, querying continuously). `agent.py`'s fallback auto-build exists so the single-command convention every other example in this repo follows (`uv run agent.py ...` and nothing else) still works if you skip `build_index.py` — but it prints a `WARNING`, not a silent success, because taking that path means the first query paid an embedding cost that didn't need to be there.
- **`--force` / `--rebuild` both delete `.chroma_crag/` before rebuilding, not just re-add documents** — Chroma has no built-in "clear and re-embed this collection" call used here; wiping the directory and starting fresh is simpler than tracking which documents changed since the last build.

## Where this goes next

This is "agentic RAG" from [Stage 5's spectrum table](../../docs/stage-05-rag-retrieval.md#the-rag-architecture-spectrum) — retrieval as a decision loop, not a single fixed pass. The table's other upgrades (hybrid keyword+vector search, re-ranking, Graph RAG) would each add a node or two to this same graph shape, not require a different architecture. If you want to see the same "grade before trusting" instinct applied to *saved facts* instead of an external corpus, see [Stage 4's forgetting/supersession policies](../../docs/stage-04-memory-state.md#forgetting-and-eviction--long-term-memory-needs-a-cleanup-policy-too) — a related but distinct problem (bad retrieval vs. stale memory).
