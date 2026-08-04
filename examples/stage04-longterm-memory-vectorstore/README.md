[← Back to roadmap](../../README.md)

# Example 02d — Long-Term Memory with a Real Vector Store (Semantic Recall)

The upgrade [`examples/stage04-longterm-memory-agent/`](../stage04-longterm-memory-agent/)'s README points at directly: that example's flat JSON fact store is replaced here with a real local vector store ([ChromaDB](https://www.trychroma.com/)), so facts are retrieved by **meaning**, not by dumping every saved fact into every prompt. This is the vector-backed `LongTermMemory` class from [Stage 4](../../docs/stage-04-memory-state.md#the-three-memory-types) and [Stage 5 — RAG + Retrieval](../../docs/stage-05-rag-retrieval.md) applied to a handful of personal facts instead of a document corpus.

## What changed from the flat-JSON version

| | Flat JSON (`stage04-longterm-memory-agent`) | This example |
|---|---|---|
| Storage | One JSON file, all facts | ChromaDB, a local embedded vector database |
| What's injected into the prompt | **Every** saved fact, every turn | Only the top `RECALL_K` (default 3) facts most relevant to the *current message* |
| Matching | Exact — you'd need the same key | Semantic — recalls a fact even if the question shares no words with how it was saved |
| Cost as facts grow | Grows with total facts ever saved | Stays flat — always retrieves at most `RECALL_K`, regardless of how many facts exist |

Same `save_fact` tool, same idea (the model decides what's worth keeping permanently) — the only thing that changed is *how recall works*.

## Setup

Everything from [`examples/stage04-longterm-memory-agent/`](../stage04-longterm-memory-agent/) (Ollama, `llama3.2:3b`, repo-root `.env`), **plus one more model** — an embedding model, pulled separately from the chat model because embedding and chat completion are different jobs:

```bash
ollama pull nomic-embed-text   # ~274MB, one-time
```

**This embedding pull is required even if `USE_LOCAL_MODEL=false`** (using the Claude API for chat). Anthropic's API doesn't offer an embeddings endpoint, so this example always embeds facts locally via Ollama, regardless of which model is answering you. That's a real asymmetry worth noticing: the *chat* model is a one-line swap (see `_build_llm()`); the *embedding* model here isn't, without more code.

## Run it

```bash
cd examples/stage04-longterm-memory-vectorstore
uv run agent.py
```

Save a fact, then ask about it using **completely different words** — this is the actual point of semantic recall, so make the wording mismatch deliberate:

```
you> we plan to start ai learning on 30th july, remember that
you> quit
```

```bash
uv run agent.py --reset      # clears the chat log only, facts survive
uv run agent.py
you> when are we kicking things off?
```

Watch stderr: `[recall] retrieved facts relevant to this message: scheduled_start_date` — it found the July fact even though "kicking things off" shares zero words with "start ai learning on 30th july." An exact-match/keyword system would miss this entirely; that's the actual capability this example adds.

Other flags, same as the flat-JSON version:

```bash
uv run agent.py --facts     # print all saved long-term facts and exit
uv run agent.py --forget    # clear long-term facts only (chat log untouched)
```

## What to look at closely

- **`LongTermMemory.remember()` deletes-then-adds on the same `key`** — [agent.py](agent.py)'s `store.get(where={"key": key})` finds any existing fact with that key and deletes it before adding the new one. Same supersession-on-write policy as the flat-JSON version's dict overwrite, just expressed as a vector-store delete+insert instead of a Python dict assignment.
- **`recall_relevant(query, k=RECALL_K)` vs. `recall_all()`** — the chat loop calls `recall_relevant()` every turn (bounded, cheap); only the `--facts` CLI flag calls `recall_all()`, because that one's a debugging tool, not something that should run on every message.
- **The system prompt says "Relevant known facts (retrieved for this message, may be empty)"** — not "here are all your facts." The model is explicitly told this list is already filtered, so it shouldn't assume a fact doesn't exist just because it's missing from a given turn's retrieval — it might just not have been relevant to *that* question.
- **`page_content=f"{key}: {value}"`** — the text that actually gets embedded is the human-readable sentence, not raw JSON. Embedding quality depends on the text reading naturally; embedding `{"key": "name", "value": "Pragnesh"}` as a literal string would embed worse than `"name: Pragnesh"` does, because the model's embedding space is trained on natural language, not JSON syntax.

## The tradeoff this doesn't remove

Semantic recall fixes "which facts get shown," but it doesn't fix the tool-call-judgment gap documented in the flat-JSON example's README: Llama 3.2 3B still occasionally re-saves a fact that's already in the retrieved "Relevant known facts" block, because recognizing "I was just told this already" requires reading and comparing against retrieved context, not just retrieving it correctly. Retrieval and judgment are different capabilities — a better retrieval system doesn't automatically produce a more careful model. Claude, tested the same way, still reliably skips the redundant save.

## Where this goes next

- **TTL / decay** — [Stage 4's forgetting section](../../docs/stage-04-memory-state.md#forgetting-and-eviction--long-term-memory-needs-a-cleanup-policy-too) still applies here and still isn't implemented: a fact saved once stays in the vector store forever until `--forget` wipes everything. Chroma's metadata filtering (already used for `where={"key": key}` above) is exactly what you'd use to add a `created_at` cutoff and an `evict_expired()` method, matching the pattern in that doc's `LongTermMemory` class.
- **A managed vector store** — Chroma here is local and embedded (a folder on disk, no server). Swapping in pgvector, Pinecone, or Weaviate is the natural step once "one person's laptop" stops being the deployment target — see [Stage 5](../../docs/stage-05-rag-retrieval.md) for the broader RAG patterns this same retrieval mechanism connects to.
