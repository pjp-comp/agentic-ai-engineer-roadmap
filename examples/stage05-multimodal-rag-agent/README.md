[← Back to roadmap](../../README.md)

# Multimodal, Agentic RAG Over Real PDFs (Text + Tables + Images)

This example combines everything the last several examples built separately: [Stage 5](../../docs/stage-05-rag-retrieval.md)'s corrective-RAG retrieval loop, a **multi-vector retrieval** technique (the current best-practice pattern for making tables and images genuinely searchable, not just present in an index), and [Stage 6](../../docs/stage-06-sessions-state.md)'s persistent Session/Event pattern so a multi-turn research conversation survives a restart.

Two source PDFs ship with this example, both indexable into **one shared store**:

| PDF | Contents |
|---|---|
| `sample.pdf` | 3 pages, 1 table, 1 chart — the simple case |
| `sample_complex.pdf` | 5 pages, 3 tables, 3 charts, with topically adjacent tables/charts (two different tables both about "% growth," two different charts both trending "over time") so retrieval and grading have to discriminate WHICH one answers a question, not just detect "a table exists somewhere" |

**Ollama only** — three local models, each doing a genuinely different job:

| Model | Job |
|---|---|
| `llama3.1:8b` | Answer generation, relevance grading, table summarization |
| `llava:7b` | **Vision model** — describes what a chart image actually depicts. This is the one step nothing else in this pipeline can substitute for: no text model can look at image bytes. |
| `nomic-embed-text` | Turns text into vectors for retrieval |

## One shared index, not one per PDF

Every PDF you index goes into the **same** `.chroma_multimodal/` directory and the **same** `docstore.json`, not a separate store per document. This is what makes "just add more PDFs later" actually work: run `build_index.py --pdf <file>` again with a different file, and its content becomes searchable *alongside* whatever's already indexed — you don't have to pick which index to query, and a question can pull an answer from multiple documents in the same turn.

What makes this safe:

- Every element's `element_id` is **namespaced by source filename** (`sample.pdf::table-8`, `sample_complex.pdf::table-8`) — so two different PDFs' "table 8" never collide and silently overwrite each other in the shared docstore.
- Every element carries a **`source` field** in its Chroma metadata — retrieval can search across every indexed PDF at once (the default) or be narrowed to one document with `--source <filename>`.
- `build_index.py` **merges** into the existing docstore rather than overwriting it — indexing a second PDF never touches the first one's entries. `--force` is the explicit escape hatch when you actually want to wipe everything and start clean.

## Why "multi-vector retrieval" — the actual technique, not just a name

A naive RAG pipeline extracts a PDF's text into one blob and embeds chunks of it. That breaks down for two of these documents' three element types:

- **A table**, embedded as its raw grid (`"Region | 2023 | 2024 | YoY\nNorthern Europe | 58.2% | 64.7% | +6.5 pp\n..."`), retrieves poorly — a question phrased in normal language ("what was East Asia's change") doesn't vector-match well against a wall of numbers and pipe characters.
- **An image**, as raw PNG bytes, has *no text representation at all*. It's completely invisible to a text-embedding vector search unless something converts it to text first.

The fix — the "multi-vector" or "parent document" pattern used by LangChain's `MultiVectorRetriever` and Unstructured.io's reference RAG architecture — is to **embed a summary, but generate from the original**:

```
ingest.py:  PDF -> Elements (text block | table | image), each tagged with
            its page AND its source filename

index.py:   for each Element:
              text  -> use directly, no model call
              table -> llama3.1:8b writes a 2-3 sentence summary of what it shows
              image -> llava:7b (VISION model) writes a 2-3 sentence description
            embed the SUMMARY -> the SHARED Chroma collection (.chroma_multimodal/)
            merge the RAW content (full table, or the image's own description)
              into the SHARED docstore.json, linked by (namespaced) element_id

agent.py:   retrieve  -- vector search over SUMMARIES across every indexed
                         PDF (or one, with --source)
            grade     -- is each retrieved summary actually useful, not just topically close?
            generate  -- build the answer from RAW content (exact numbers, not
                         a lossy summary of them), pulled from docstore via element_id,
                         cited with BOTH source document and page
```

Retrieval quality comes from the summary (phrased the way a question would be); answer *accuracy* comes from the original content never having been lossily compressed before being handed to the model that writes the final answer. Conflating these two — embedding and answering from the same text — is the mistake this technique specifically avoids.

## The graph

```
retrieve (vector search over summaries, across all indexed PDFs or one, via --source)
   -> grade (per-candidate: actually useful, or just topically adjacent?)
        -- relevant found --------> generate (from RAW content, cited by source+page) -> answer
        -- nothing relevant, ------> rewrite_query -> retrieve (loop back, once)
           haven't retried
        -- nothing relevant, ------> refuse
           already retried
```

Same shape as [`stage05-rag-langgraph`](../stage05-rag-langgraph/)'s corrective RAG — this file expresses it as a plain function chain instead of a `StateGraph`, since the point here is the multi-vector technique, the shared index, and the session integration, not re-demonstrating LangGraph's node/edge machinery a second time.

## Sessions: every question is a turn in a resumable research conversation

Every call to `agent.py` is a turn recorded into a [`PersistentSessionService`](session.py) — the same `Event`/`Session` pattern as [`stage06-sessions-plain`](../stage06-sessions-plain/), copied into this example's own `session.py` (each `examples/*` directory here is a standalone project, so it's a deliberate copy, not a cross-directory import).

Three ways to continue a session:

- **`--resume <session_id>`** — one question, one new process, same session. Fine for scripting; tedious for a real back-and-forth (re-pays startup cost and requires retyping the session id every call).
- **`--chat [session_id]`** — an interactive loop. Loads the vector store, docstore, and session **once**, then every line you type becomes another turn against the same session in the same process — no restart, no retyping an id, exactly like a real chat UI.
- **`--history <session_id>` / `--list`** — inspect a session's full state and event log, or list every session on disk, same as [`stage06-sessions-plain`](../stage06-sessions-plain/).

## Run it

```bash
cd examples/stage05-multimodal-rag-agent
ollama pull llama3.1:8b        # one-time, ~4.9GB
ollama pull llava:7b            # one-time, ~4.7GB, vision model
ollama pull nomic-embed-text    # one-time, ~274MB, shared with other examples
uv run build_index.py                          # index sample.pdf into the shared store
uv run build_index.py --pdf sample_complex.pdf  # ADD the complex doc alongside it
```

```
Parsing sample.pdf...
  1 image, 14 text, 1 table
Summarizing + embedding elements (tables/images need a model call each, this takes a minute)...
Done. .chroma_multimodal/ now contains 1 document(s): sample.pdf

Parsing sample_complex.pdf...
  3 image, 27 text, 3 table
Summarizing + embedding elements (tables/images need a model call each, this takes a minute)...
Done. .chroma_multimodal/ now contains 2 document(s): sample.pdf, sample_complex.pdf
```

Ask a question that only ONE of the two documents can answer — the shared index finds it without you saying which document to look in:

```bash
uv run agent.py "How much did battery storage cost decline?"
```

```
  [retrieve] query="..." -> 4 candidate(s): text(sample_complex.pdf p3), text(sample_complex.pdf p3), image(sample_complex.pdf p3), text(sample_complex.pdf p3)
  [grade] sample_complex.pdf::text-19 (text, p3): relevant
  ...

According to sample_complex.pdf, page 3, battery storage cost declined by 65% from $280/kWh in 2019 to $98/kWh in 2024. (sample_complex.pdf, page 3)
```

Ask a question BOTH documents partially answer — watch the citation name two different sources in one response:

```bash
uv run agent.py "What was East Asia's year-over-year change in renewable adoption?"
```

```
East Asia recorded a 7.7 percentage point year-over-year gain in renewable adoption (sample.pdf, page 2). This gain was driven primarily by accelerated solar deployment (sample_complex.pdf, page 2).
```

Narrow retrieval to just one document with `--source` — useful when you know which report an answer should come from, and a good way to see the refusal path even on a topic the OTHER document covers:

```bash
uv run agent.py --source sample.pdf "How much did battery storage cost decline?"
```

`sample.pdf` genuinely has no storage-cost content, so this correctly retries once, still finds nothing, and refuses — proof `--source` actually restricts retrieval rather than just being cosmetic.

Continue a session interactively instead of one `--resume` call at a time:

```bash
uv run agent.py --chat
```

```
Chatting about all indexed documents -- session 7ac25bce-...
Type a question, or 'exit'/'quit' to stop. Ctrl-C also works.

> What was East Asia's year-over-year change?
...
East Asia's year-over-year change was 7.7 percentage points (sample.pdf, page 2).

> Which grew faster, solar or wind?
...
According to sample.pdf, solar capacity overtook wind for the first time in 2024...

> exit
Session 7ac25bce-... saved. Resume it later: uv run agent.py --chat 7ac25bce-...
```

Force the refusal path with a question neither document covers:

```bash
uv run agent.py "What is the capital of France?"
```

## Real bugs this surfaced, and the actual fixes

Three genuine failures came up during development, all fixed in the code, all worth knowing about because they generalize past this one example:

**1. Table text was being extracted twice.** The first version of `ingest.py`'s text/table separation used a length heuristic (drop any text "paragraph" shorter than 40 characters, on the theory that table rows are short). This silently failed: pymupdf's plain text extraction returns each table row as its own line, and real rows like `"Northern Europe   58.2%   64.7%   +6.5 pp"` are *longer* than 40 characters, so they passed the filter and got embedded a second time as garbled text elements alongside the properly-extracted table element. The fix: extract text as **blocks with real bounding boxes**, and drop any block whose bbox geometrically intersects a detected table's bbox — geometry, not a length guess, is what correctly separates "this text is the table" from "this text is a normal paragraph." See `ingest.py`'s `parse_pdf()`.

**2. The grader was too lenient on bare section headings.** Asking "What is the capital of France?" initially returned a confused non-answer instead of a clean refusal, because a 12-character summary ("1. Overview") got graded "relevant" — with almost no content to reason over, the small model tended to guess yes rather than commit to no. Two fixes: `grade()` now skips anything under 30 characters outright (too short to plausibly answer anything on its own), and the grading prompt was tightened from "discusses the same topic" to "contains information that's ACTUALLY useful... not just a vaguely related topic or section title." As a second layer of defense, `generate()`'s prompt was also hardened to refuse plainly rather than comment on the context itself if what it's given genuinely doesn't answer the question — a system shouldn't rely on grading catching every weak case perfectly.

**3. First version indexed each PDF into its own isolated directory.** Adding a second PDF meant a second, disconnected `.chroma_multimodal_<name>/` store, and querying required knowing in advance which one to open — the opposite of "just add more PDFs." Fixed by moving to one shared collection with a `source` field on every element (see "One shared index, not one per PDF" above) — the same fix that makes `--source` filtering possible at all.

## What to look at closely

- **`grade()` grades the SUMMARY, not the raw content** — cheap and fast, since pulling full raw content (a whole table, or re-describing an image) for every one of the k candidates, most of which won't pass grading anyway, would be wasted work. Only elements that pass get their raw content attached, in `grade()`'s final step.
- **`generate()` explicitly uses `c["raw"]`, never `c["summary"]`** — this is the entire point of the multi-vector split made concrete in code: the vector that got matched during search is deliberately a different string from what gets shown to the answer-writing model.
- **`retrieve()`'s `source_filter` becomes a Chroma metadata `filter={"source": ...}`** — this is a real, structural narrowing of the search space, not a post-hoc filter applied to results after the fact; Chroma only considers vectors matching that filter in the first place.
- **Small images get filtered out during ingestion** (`ingest.py`, `len(image_bytes) < 5000`) — page decorations, bullet icons, and logos aren't worth a vision-model call and would only add retrieval noise. A genuinely important but small diagram could theoretically get skipped by this threshold, worth knowing if you point this at your own PDF.
- **The vision model isn't perfect, and that's shown honestly, not hidden** — `llava:7b`'s chart descriptions occasionally get exact details (years, axis labels) slightly imprecise while correctly identifying the overall trend. The pipeline doesn't correct or verify this description against ground truth; it's trusted as-is, the same way any RAG system trusts its summarization step.
- **`--chat` and `--resume` share `_answer_turn()` and `_load_session()`** rather than duplicating turn logic — resuming a session isn't a special code path, it's the same turn function applied to whichever `Session` object got loaded, same principle as [`stage06-sessions-plain`](../stage06-sessions-plain/)'s `run_session()`.

## Where this goes next

This is [Stage 5](../../docs/stage-05-rag-retrieval.md)'s spectrum table's "agentic RAG" row, [Stage 6](../../docs/stage-06-sessions-state.md)'s session/event persistence, and multimodal ingestion over a growing document collection, combined into one pipeline. Compare against [`stage05-rag-langgraph`](../stage05-rag-langgraph/) for the same corrective-RAG logic expressed as an actual `StateGraph` instead of a function chain, and [`stage06-sessions-plain`](../stage06-sessions-plain/) for the session mechanics on their own, without retrieval layered on top. To add your own PDF: drop it in this directory and run `uv run build_index.py --pdf your_file.pdf`.
