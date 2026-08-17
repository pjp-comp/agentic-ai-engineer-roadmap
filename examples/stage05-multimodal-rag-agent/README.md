[← Back to roadmap](../../README.md)

# Multimodal, Agentic RAG Over Real PDFs (Text + Tables + Images)

This example combines [Stage 5](../../docs/stage-05-rag-retrieval.md)'s corrective-RAG retrieval loop, a **multi-vector retrieval** technique (the current best-practice pattern for making tables and images genuinely searchable, not just present in an index), and [Stage 6](../../docs/stage-06-sessions-state.md)'s persistent Session/Event pattern so a multi-turn research conversation survives a restart.

**Bring your own files.** This example ships no sample document and no PDF generator — drop whatever you want to query into `assets/` (PDFs, `.txt`/`.md`, or standalone images), then run one command to index everything in there. Everything scales to as many files as you add, all searchable together from one shared index. `assets/*` is gitignored (except a `.gitkeep` that keeps the empty folder itself tracked), so whatever you drop in there stays local and is never accidentally committed.

**Ollama only** — three local models, each doing a genuinely different job:

| Model | Job |
|---|---|
| `llama3.1:8b` | Answer generation, relevance grading, table summarization |
| `qwen2.5vl:7b` | **Vision model** — describes what a chart image actually depicts. See "Why not CLIP" below for why this has to be a vision-*language* model, not a vision-*similarity* model. |
| `nomic-embed-text` | Turns text into vectors for retrieval |

## One file per pipeline stage

Each stage of the pipeline is its own module, so you can read (or modify) any one part without the rest:

| File | Stage | What it owns |
|---|---|---|
| [`chunking.py`](chunking.py) | **Chunking** | `parse_file()` dispatches by extension — `parse_pdf()` (element-aware: text/table/image), `parse_text_file()` (`.txt`/`.md`, paragraph-chunked), `parse_image_file()` (standalone `.png`/`.jpg`/`.jpeg`/`.webp`, one image `Element`) — every path produces the same typed `Element`s, tagged with page + source filename |
| [`vision.py`](vision.py) | **Summarization** | `summarize_elements()` — each `Element` becomes a short text summary (tables/images need a model call; text passes through). Duplicate images (byte-identical, e.g. a repeated letterhead logo) are described once and reused — see bug #8 |
| [`embeddings.py`](embeddings.py) | **Embeddings** | Which model turns text into vectors — `nomic-embed-text` by default, an optional CLIP path documented but off |
| [`vector_store.py`](vector_store.py) | **Vector store** | Owns the shared Chroma collection (`.chroma_multimodal/`) and `docstore.json` (raw content, looked up by `element_id`) |
| [`keyword_search.py`](keyword_search.py) | **Keyword search** | `keyword_search()` — BM25 exact-term search over the same summaries, for hybrid retrieval |
| [`retrieval.py`](retrieval.py) | **Retrieval** | `retrieve()` (vector search), `hybrid_retrieve()` (vector + BM25, fused by rank), `grade()` (is this candidate actually useful?), `rewrite_query()` (retry once) |
| [`generation.py`](generation.py) | **Generation** | `generate()` — builds the final answer from raw content, including a real fix for table-sum arithmetic |
| [`pipeline.py`](pipeline.py) | **Pipeline** | Wires retrieval + generation into the corrective-RAG loop: `retrieve → grade → (generate \| rewrite_query → retrieve again \| refuse)` |
| [`session.py`](session.py) | **Sessions** | `Event`/`Session`/`PersistentSessionService` — same pattern as [`stage06-sessions-plain`](../stage06-sessions-plain/) |
| [`build_index.py`](build_index.py) | **Indexing CLI** | Scans `assets/` and indexes every supported file in ONE command — chunking → summarization → vector store, per file, skipping whatever's already indexed |
| [`agent.py`](agent.py) | **Query CLI** | Thin — argument parsing, session load/resume, calls `pipeline.run_rag_turn()` per question |

`retrieval.py` and `generation.py` never import each other — `pipeline.py` is the only file that knows about both, so each stays independently callable (e.g. `from retrieval import retrieve, grade` to inspect just the search step without running a full turn).

## Data flow, end to end

```
assets/your_file.pdf
   │
   ▼  chunking.parse_pdf()
[Element(kind="text"), Element(kind="table"), Element(kind="image"), ...]
   │
   ▼  vision.summarize_elements()
   │    text  -> used as-is, no model call
   │    table -> llama3.1:8b writes a 2-3 sentence summary
   │    image -> qwen2.5vl:7b describes every category/value it can read
   ▼
[{element_id, kind, page, source, summary, raw}, ...]
   │
   ▼  vector_store.build_multi_vector_index()
   │    embed each SUMMARY  -> Chroma (.chroma_multimodal/, ONE SHARED store)
   │    merge each RAW      -> docstore.json (keyed by element_id)
   ▼
════════════════════════════ index built, query time ════════════════════════════
   │
   ▼  retrieval.retrieve(question)              -- vector search over SUMMARIES (default)
   │  or retrieval.hybrid_retrieve(question)     -- vector search + keyword_search.keyword_search()
   │       (BM25), fused by Reciprocal Rank Fusion -- opt in with --hybrid
   ▼  retrieval.grade(candidates, question)  -- per-candidate: actually useful?
   │    relevant found        -> generation.generate(raw content) -> answer
   │    nothing, first try    -> retrieval.rewrite_query() -> retrieve again
   │    nothing, already retried -> generation.refuse()
   ▼
answer, recorded as a turn in a persisted session.py Session
```

## Why "multi-vector retrieval" — the actual technique, not just a name

A naive RAG pipeline extracts a PDF's text into one blob and embeds chunks of it. That breaks down for two element types:

- **A table**, embedded as its raw grid (`"Region | 2023 | 2024 | YoY\nNorthern Europe | 58.2% | 64.7% | +6.5 pp\n..."`), retrieves poorly — a question phrased in normal language doesn't vector-match well against a wall of numbers and pipe characters.
- **An image**, as raw PNG bytes, has *no text representation at all*. It's completely invisible to a text-embedding vector search unless something converts it to text first.

The fix — the "multi-vector" or "parent document" pattern used by LangChain's `MultiVectorRetriever` and Unstructured.io's reference RAG architecture — is to **embed a summary, but generate from the original**. Retrieval quality comes from the summary (phrased the way a question would be); answer *accuracy* comes from the original content never having been lossily compressed before being handed to the model that writes the final answer. `retrieval.grade()` only attaches `raw` content to candidates that pass grading; `generation.generate()` reads `raw`, never `summary`.

## Hybrid RAG — vector search fused with keyword search

Vector search alone has a real, known weakness: embeddings are lossy for **exact tokens**. A model number, a ticker symbol, a precise figure, or a region name spelled a specific way can sit at merely "pretty close" in vector space to a paraphrase that doesn't actually contain it, while a chunk that literally contains the exact string can rank behind it. This is docs/stage-05-rag-retrieval.md's own "Advanced/Hybrid" row in the RAG spectrum table — "pure semantic search misses exact terms like ticker symbols" — implemented here for real, not just described.

**How it works**, opt in with `--hybrid`:

1. `retrieval.retrieve()` runs vector search over summaries — same as always, paraphrase-tolerant.
2. `keyword_search.keyword_search()` runs **BM25** (the ranking algorithm behind Elasticsearch/OpenSearch's default full-text scoring, via the `rank-bm25` package) over the same summaries — exact-term-precise, zero notion of meaning.
3. `retrieval.reciprocal_rank_fusion()` merges the two ranked lists into one. **By rank position, not raw score** — a Chroma cosine-similarity score and a BM25 score are on incomparable scales, so RRF sidesteps normalizing them at all: `score(doc) = Σ 1 / (60 + rank_in_each_list)`. A candidate that ranks well in *both* lists — relevant by two independent measures — rises to the top; a candidate only one method found still gets a chance, just weighted lower. This is the same fusion technique Elasticsearch, Azure AI Search, and Weaviate's hybrid mode all use.
4. The fused list flows into `grade()`/`generate()` exactly like a vector-only result set — neither downstream stage needs to know which retrieval mode produced its input.

```bash
uv run agent.py --hybrid "a question with an exact term, figure, or name in it"
```

**A real, honest finding from testing this at small scale**: BM25's IDF formula gives near-zero (and, in `rank_bm25`'s implementation, exactly-zero-floored) weight to a term that appears in *more than half* the indexed documents — by design, since a term that common isn't discriminating between documents in a large corpus. At this example's scale (a handful of PDFs), a product code or region name mentioned in 2 of 4 indexed chunks can trigger that floor and contribute nothing to the keyword side of the fusion, even though the term is a perfectly good exact match. This isn't a bug in `keyword_search.py` — it's genuine BM25 behavior, verified by inspecting `rank_bm25`'s own computed IDF table directly — but it's worth knowing that hybrid's keyword advantage is most visible at real scale (hundreds+ of chunks), not on a 4-document test corpus where vector search alone often already finds everything.

`retrieve()` (vector-only) is kept as its own separate, still-callable function — hybrid isn't presented as a strict upgrade that replaces it, it's an additional mode you opt into, specifically so the two can be compared on the same query if you want to see the difference for yourself.

## One shared index, not one per PDF

Every file you index goes into the **same** `.chroma_multimodal/` directory and the **same** `docstore.json`. Drop a new file into `assets/` and re-run `build_index.py` (no arguments needed), and its content becomes searchable *alongside* whatever's already indexed — a question can pull an answer from multiple documents in the same turn.

What makes this safe:

- Every element's `element_id` is **namespaced by source filename** (`report_a.pdf::table-8`, `report_b.pdf::table-8`) — two different files' "table 8" never collide in the shared docstore.
- Every element carries a **`source` field** in its Chroma metadata — retrieval searches across every indexed file by default, or narrows to one with `--source <filename>`.
- `vector_store.build_multi_vector_index()` **merges** into the existing docstore rather than overwriting it. `build_index.py --force` is the explicit escape hatch when you actually want to wipe everything and start clean.

## More than PDFs — `.txt`/`.md` and standalone images

`assets/` doesn't have to be all PDFs. `chunking.parse_file()` dispatches by file extension:

| Extension | Parser | What happens |
|---|---|---|
| `.pdf` | `parse_pdf()` | Full element-aware extraction — text blocks, tables, embedded images, each with page numbers and (for images) caption grounding from nearby text |
| `.txt`, `.md` | `parse_text_file()` | No parsing library, no model call — the file's own text, split into paragraph-sized chunks (blank-line-separated) so it embeds at the same granularity a PDF's per-block text does, rather than as one giant blob |
| `.png`, `.jpg`, `.jpeg`, `.webp` | `parse_image_file()` | The whole file becomes ONE image `Element`, sent straight to `vision.summarize_image()` — same vision-model path a chart embedded inside a PDF takes, just with no surrounding page text to draw a caption from (`caption=""`) |

Every path produces the same `Element` shape (`kind`, `page`, `content`/`image_bytes`, `source`, `element_id`) regardless of source format, so nothing downstream of `chunking.py` — summarization, embedding, retrieval, generation — needs to know or care which parser produced a given element. `page` is always `1` for `.txt`/`.md`/image files (page numbers are a PDF-specific citation detail that doesn't exist for those formats).

An unsupported extension (`.docx`, `.csv`, anything else) is a **loud error** when explicitly targeted with `--file`/`--pdf`, and silently excluded (not an error) when scanning `assets/` with no arguments — dropping a file type this pipeline doesn't handle yet into `assets/` won't crash your next `build_index.py` run, but trying to index it directly will tell you clearly that it isn't supported, rather than silently producing an empty or wrong index entry.

## Why not CLIP — a real evaluation, not a guess

CLIP embeds images and text into the *same* vector space, so it's a reasonable question: why not embed chart images directly with CLIP instead of running them through a vision-language model first?

Because CLIP and a vision-language model (like `qwen2.5vl:7b`) do fundamentally different jobs. **CLIP is trained to match natural photos against short captions** — "a photo of a dog on a beach" — it's excellent at broad visual/semantic similarity: find images that *look like* a description. It has no OCR-like capability and no concept of "read the axis labels and bar heights." Embedding a bar chart with CLIP would tell you "this looks like a chart," not what the chart *says* — CLIP cannot extract that Manufacturing added 420,000 jobs, because it was never trained to read text or values out of an image at all.

This isn't theoretical for this example — it's the exact problem "Real bugs this surfaced" bug #4 below documents solving. The fix that made chart-reading actually work was swapping to a **stronger vision-language model**, not changing the embedding strategy. CLIP addresses a different, genuinely useful problem this pipeline doesn't currently have ("find images that look similar to this one" — reverse image search) — not a substitute for reading a chart's content.

`embeddings.py` ships a working, opt-in `ClipImageEmbedder` for that different use case (image-to-image or image-to-text similarity search over a picture collection), off by default (`USE_CLIP = False`), with the `torch`/`transformers` dependency it needs kept out of the default install (`uv pip install -e '.[clip]'` to add it). It is not wired into the indexing pipeline as a replacement for `vision.summarize_image()` — see `embeddings.py`'s module docstring for the full reasoning.

## Caption grounding — how images get described accurately

A vision model summarizing a chart from pixels alone is prone to hallucination — inventing numbers, category names, or trends that aren't actually in the image. This example defends against that with **caption grounding**: every image `Element` carries a `caption` field, populated in `chunking.py` by taking the image's own bounding box (`page.get_image_rects()`) and collecting only the text blocks within a tight vertical window (`CAPTION_WINDOW_PT = 80` points, roughly one paragraph) above and below it — not the whole page's text.

That narrow window matters concretely: an early version of this fix used the *entire page's* text as context. On a page with both a chart AND an unrelated table below it, that caused the vision model in use at the time to pull numbers from the wrong table and misattribute them as the chart's own category values — a worse failure than having no caption at all. Narrowing to a tight window around the image's own position on the page fixed that specific failure mode.

`vision.summarize_image()` passes this narrow caption to the vision model alongside the image bytes, so a chart's headline figure stated in words nearby ("Installation added 680,000 jobs...") can corroborate the model's own reading of the chart. Caption grounding is a real, independent improvement — but as "Real bugs this surfaced" below covers, it was not sufficient on its own to fix accurate reading of *every* bar's value; the vision model itself mattered more.

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
ollama pull qwen2.5vl:7b        # one-time, ~6GB, vision model
ollama pull nomic-embed-text    # one-time, ~274MB, shared with other examples

# drop as many files as you want into assets/ (PDFs, .txt/.md, images),
# then ONE command indexes everything:
uv run build_index.py
```

```
sample.pdf: parsing...
  1 image, 14 text, 1 table
  Summarizing + embedding (tables/images need a model call each, this takes a minute)...
notes.txt: parsing...
  4 text
  Summarizing + embedding (tables/images need a model call each, this takes a minute)...
chart.png: parsing...
  1 image
  Summarizing + embedding (tables/images need a model call each, this takes a minute)...

Done. .chroma_multimodal/ now contains 3 document(s): chart.png, notes.txt, sample.pdf
```

Drop in a 4th file later and re-run the exact same command — only the new file gets processed, everything already indexed is skipped:

```bash
uv run build_index.py
```

```
sample.pdf: already indexed, skipping (use --force to rebuild everything)
notes.txt: already indexed, skipping (use --force to rebuild everything)
chart.png: already indexed, skipping (use --force to rebuild everything)
new_report.pdf: parsing...
  ...

Done. .chroma_multimodal/ now contains 4 document(s): chart.png, new_report.pdf, notes.txt, sample.pdf
```

**See exactly what the vision model wrote for every indexed image** — which image got which description, in one place, without running a query and working backward from an answer:

```bash
uv run build_index.py --describe-images
```

```
4 indexed image(s):

=== sample.pdf, page 1 (sample.pdf::image-1) ===
The image is a line graph titled "Solar vs. Wind Adoption, 2019-2024." ...
  - 2019: Solar 3.8%, Wind 6.2%
  - 2020: Solar 5.7%, Wind 7.4%
  ...

=== sample_complex.pdf, page 4 (sample_complex.pdf::image-24) ===
The image is a horizontal bar chart titled "Renewable Sector Job Growth by Category, 2024." ...
  - Manufacturing: Approximately 420 thousand jobs
  - Installation: Approximately 680 thousand jobs (the largest category)
  ...
```

This IS the exact text that got embedded and would be used to answer a question about that image (see "Why not CLIP" and "Caption grounding" above) — if a chart-based answer looks wrong, this is the first place to look, before assuming retrieval or grading is the problem. Narrow it to one file with `--source`:

```bash
uv run build_index.py --describe-images --source sample.pdf
```

Ask a question:

```bash
uv run agent.py "what does the report say about X?"
```

Narrow retrieval to just one document with `--source`, when you know which report an answer should come from:

```bash
uv run agent.py --source your_file.pdf "..."
```

Use hybrid retrieval (vector + BM25 keyword search, fused) for a question with exact terms/figures in it — composes with `--source`, `--resume`, and `--chat`:

```bash
uv run agent.py --hybrid "a question with an exact term or figure in it"
```

Continue a session interactively instead of one `--resume` call at a time:

```bash
uv run agent.py --chat
```

```
Chatting about all indexed documents -- session 7ac25bce-...
Type a question, or 'exit'/'quit' to stop. Ctrl-C also works.

> a question
...
an answer

> a follow-up
...
another answer

> exit
Session 7ac25bce-... saved. Resume it later: uv run agent.py --chat 7ac25bce-...
```

`--history <session_id>` and `--list` work as described above.

## Real bugs this surfaced, and the actual fixes

Eight genuine failures came up during development, all fixed (or, for #7, honestly documented as expected behavior rather than a bug) in the code:

**1. Table text was being extracted twice.** The first version of `chunking.py`'s text/table separation used a length heuristic (drop any text "paragraph" shorter than 40 characters, on the theory that table rows are short). This silently failed: pymupdf's plain text extraction returns each table row as its own line, and real rows can be *longer* than 40 characters, so they passed the filter and got embedded a second time as garbled text elements alongside the properly-extracted table element. The fix: extract text as **blocks with real bounding boxes**, and drop any block whose bbox geometrically intersects a detected table's bbox — geometry, not a length guess, is what correctly separates "this text is the table" from "this text is a normal paragraph." See `chunking.py`'s `parse_pdf()`.

**2. The grader was too lenient on bare section headings.** A generic off-topic question initially returned a confused non-answer instead of a clean refusal, because a 12-character summary ("1. Overview") got graded "relevant" — with almost no content to reason over, the small model tended to guess yes rather than commit to no. Two fixes: `retrieval.grade()` now skips anything under 30 characters outright (too short to plausibly answer anything on its own), and the grading prompt was tightened from "discusses the same topic" to "contains information that's ACTUALLY useful... not just a vaguely related topic or section title." As a second layer of defense, `generation.generate()`'s prompt was also hardened to refuse plainly rather than comment on the context itself if what it's given genuinely doesn't answer the question — a system shouldn't rely on grading catching every weak case perfectly.

**3. First version indexed each PDF into its own isolated directory.** Adding a second PDF meant a second, disconnected store, and querying required knowing in advance which one to open — the opposite of "just add more PDFs." Fixed by moving to one shared collection with a `source` field on every element (see "One shared index, not one per PDF" above) — the same fix that makes `--source` filtering possible at all.

**4. The vision model (originally `llava:7b`) could not reliably read a multi-bar chart's individual values — fixed by swapping the model, not by more prompt tuning.** Asking a question whose answer was a *minor* bar's specific value on a 5-category chart returned a refusal, traced down to the vision-model summary itself being wrong. The investigation, in order:

  1. The chart was completely legible to a human — a simple horizontal bar chart, 5 clearly labeled bars.
  2. `llava:7b`'s summary of the full chart invented a multi-year trend narrative on what was actually single-year data, wrong/merged category names, and a fabricated total figure appearing nowhere in the real data. This is what motivated the caption-grounding fix described above.
  3. Even asked the simplest possible isolated question — *"list only the category names on the vertical axis, top to bottom, nothing else"* — `llava:7b` returned a list mixing a handful of real category names with a dozen fabricated ones, which ruled out "the summarization prompt is too complex" as the explanation.
  4. Tightening the prompt to explicitly demand every category and value made fabrication measurably **worse** on `llava:7b`, not better — more invented content to satisfy the completeness requirement, not more accurate reading.
  5. Swapping `VISION_MODEL` (in `vision.py`) to `qwen2.5vl:7b` (no other code change) and re-running the exact same isolated test returned all real categories with all correct values on the first try.

  The lesson: this was a genuine model-capability gap (structured chart/plot reading), not a prompting problem — no amount of instruction tuning fixed it on the weaker model, and almost none was needed once the model itself was strong enough. Caption grounding (kept regardless of vision model) is still a real, independent improvement for numbers that are *also* stated in nearby text; it was just never going to be sufficient on its own for a number that exists only inside the image. See "Why not CLIP" above for the related, adjacent question this also answers.

**5. `generation.generate()` did multi-row arithmetic in its head and got it wrong, even with completely correct raw data in front of it.** Asked for a total across a 4-row table column, `llama3.1:8b` produced an arithmetically wrong sum, and its own follow-up "justification," when pushed, visibly computed the correct total in one place and then contradicted itself with a different, unexplained number. This was never a retrieval or grading problem — the exact right table, with the exact right numbers, was already in context. The model's free-text arithmetic itself was unreliable, a well-documented weakness of small LLMs doing multi-step addition as generated text rather than a real calculation.

  The fix is **not** "ask the model to be more careful" (prompting cannot fix unreliable mental arithmetic) and not a full tool-calling loop (more architecture than one failure mode needs). It's `generation.compute_table_totals()`: when a graded-relevant element is a table, its raw pipe-delimited text is parsed back into columns in Python, and every column that's genuinely numeric (excluding rate/ratio columns — see below) gets summed with real arithmetic. The result is appended to that table's context block as a labeled fact, and `generate()`'s prompt explicitly instructs the model to use that number exactly rather than recompute it. The model's job becomes *stating* a fact, never *calculating* one — the same principle as the multi-vector split itself, applied to arithmetic instead of retrieval.

  One real subtlety this caught: **not every numeric-looking column should be summed.** A table with a "YoY Change," "Growth," or "Share" column full of percentages would produce a number that parses fine as arithmetic but means nothing if summed (percentages of different bases don't add up to a meaningful combined percentage). `compute_table_totals()` explicitly skips any column whose header contains `%`, "growth," "change," or "share" — this exclusion is load-bearing, not defensive over-engineering.

**6. The same question, asked repeatedly with no code change, sometimes graded the correct table "relevant" and sometimes didn't — because `temperature` was never set on any of the pipeline's `ollama.chat()` calls.** Discovered while re-verifying the arithmetic fix above: asking the same investment-total question 5 times in a row produced the correct answer most times, and once produced a fabricated total sourced from a text passage that never states any total at all — because that one run happened to grade the real table "not relevant," a call that should be a consistent binary judgment behaving nondeterministically between otherwise-identical runs. The fix: every `ollama.chat()` call across `retrieval.py` and `generation.py` now passes `options={"temperature": 0}`. Re-running the same question 5 times afterward produced the identical correct answer all 5 times. Grading and generation are judgment/extraction tasks, not creative writing — they should behave the same way on the same input every time.

**7. BM25 (`keyword_search.py`) contributed zero score for an exact product-code match on a small test corpus — verified as correct library behavior, not a bug, and documented rather than "fixed."** Testing hybrid retrieval with a query containing an exact term that appeared in 2 of 4 indexed chunks, `keyword_search()` returned 0 candidates. Direct inspection of `rank_bm25`'s internal IDF table confirmed the term's IDF was floored to exactly `0.0` — standard BM25 behavior: a term appearing in more than half a corpus is treated as too common to be discriminating, and `rank_bm25` clamps the resulting negative IDF to zero rather than letting it produce a negative relevance score. This is correct on a real-scale corpus (hundreds+ of documents), where a specific term appearing in only 2 of them would legitimately score highly — it's an artifact of testing against a handful of documents, not a defect in `build_bm25_index()`/`keyword_search()`. See "Hybrid RAG" above for what this means practically: hybrid's keyword-matching advantage is most visible at real scale, not on a tiny corpus where vector search alone often already finds everything.

**8. A real multi-page regulatory filing re-described the identical letterhead logo on every single page — the same vision-model call, repeated dozens of times, for zero new information.** A 569-element PDF with 19 images logged near-identical descriptions ("a header for [Company], includes contact information...") for consecutive images on consecutive pages — the same logo, appearing in the header of every page, each triggering its own full `qwen2.5vl:7b` call. This wasn't wrong (each description was individually accurate), it was wasteful: real indexing time spent describing pixels the pipeline had already described moments earlier.

  The fix: `vision.summarize_elements()` now hashes each image's raw bytes (SHA-256) before calling the vision model, and keeps a `hash -> summary` map for the current indexing run. The FIRST occurrence of a given hash gets a real vision-model call; every later element with the *same* hash reuses that description directly, logged as `"duplicate image (matches an earlier one) -- reusing its description, no model call"` instead of triggering another API round-trip. A byte-identical hash match is a genuine duplicate (a repeated logo really is the same PNG bytes every time it's embedded), not a heuristic guess — this is exact-match dedup, not "looks similar enough." Each occurrence still gets its own `element_id`, its own page number, and its own entry in the vector store and docstore, so retrieval and per-page citation work exactly as before — only the redundant *model call* is eliminated, not the element itself. Verified on a 5-page test PDF with the same logo on every page: only page 1's occurrence triggered a real call; pages 2-5 all correctly reused it.

## What to look at closely

- **`retrieval.grade()` grades the SUMMARY, not the raw content** — cheap and fast, since pulling full raw content for every candidate, most of which won't pass grading anyway, would be wasted work. Only elements that pass get their raw content attached.
- **`generation.generate()` explicitly uses `raw`, never `summary`** — this is the entire point of the multi-vector split made concrete in code: the vector that got matched during search is deliberately a different string from what gets shown to the answer-writing model.
- **`retrieve()`'s `source_filter` becomes a Chroma metadata `filter={"source": ...}`** — a real, structural narrowing of the search space, not a post-hoc filter applied after the fact; Chroma only considers vectors matching that filter in the first place.
- **Small images get filtered out during chunking** (`chunking.py`, `MIN_IMAGE_BYTES = 5000`) — page decorations, bullet icons, and logos aren't worth a vision-model call and would only add retrieval noise. A genuinely important but small diagram could theoretically get skipped by this threshold, worth knowing if you point this at your own PDF.
- **Caption text is windowed by vertical position, not by paragraph count** (`chunking.py`'s `CAPTION_WINDOW_PT = 80`) — a fixed distance in PDF points rather than "the paragraph before and after," which is simpler but means a very short paragraph right next to the image and a very long one just outside the window get treated inconsistently.
- **`VISION_MODEL` is a one-line swap in `vision.py`, and it's the single highest-leverage lever in this whole pipeline for image-heavy documents** — see bug #4 above for the concrete before/after. If you point this at your own PDF and a chart-based answer looks wrong, trying a different `VISION_MODEL` before touching any prompt is the first thing worth doing.
- **`build_index.py --describe-images` reads straight from `docstore.json`, no model call, no re-indexing** — it's a pure read of what's already there, safe to run anytime, as often as you want, on an index of any size. Worth running right after any `build_index.py` call on a new image-heavy file, before you've asked a single question, just to sanity-check the descriptions look right.
- **`compute_table_totals()` only fires for `kind == "table"` elements, never for text or images** — arithmetic errors from a chart's numbers (read by the vision model, not parsed as structured data) aren't fixed by this; only genuinely tabular, pipe-delimited raw content can be reliably parsed back into columns this way.
- **The numeric-column requirement is "every row parses as a number," not "most rows"** — a column with one stray text cell is deliberately excluded from summing entirely, rather than silently summing only the numeric rows and producing a total that looks complete but quietly skipped one region. Wrong is worse than absent here.
- **`--chat` and `--resume` share `_answer_turn()` and `_load_session()`** rather than duplicating turn logic — resuming a session isn't a special code path, it's the same turn function applied to whichever `Session` object got loaded, same principle as [`stage06-sessions-plain`](../stage06-sessions-plain/)'s `run_session()`.
- **`ClipImageEmbedder` in `embeddings.py` is fully working code, not a stub** — instantiate it and call `.embed_image()`/`.embed_text()` directly if you want to experiment with image-similarity search; it's just not wired into `vector_store.py`'s indexing path by default. See "Why not CLIP" above.
- **`keyword_search.py` rebuilds its BM25 index fresh on every call, from whatever's currently in the docstore** — no separate on-disk keyword index to keep in sync with `vector_store.py`'s Chroma collection. Fast enough at this example's scale that a persisted BM25 index would be complexity without a real payoff; a production system with thousands of documents would persist one instead.
- **`reciprocal_rank_fusion()` takes a list of ranked lists, not just two** — written generically (`ranked_lists: list[list[dict]]`) even though `hybrid_retrieve()` only ever passes two (vector + keyword). A third retrieval signal (e.g. the CLIP image-similarity path in `embeddings.py`) could be fused in the same way without changing the fusion function itself.
- **Image dedup hashes RAW BYTES, not the vision model's description** — two visually similar-but-not-identical images (e.g. the same chart re-rendered at a slightly different size) will NOT dedupe and will each get their own real vision-model call. This is deliberate: an exact byte hash is a guaranteed genuine duplicate; anything looser (perceptual hashing, embedding similarity) risks merging two images that only *look* alike but actually show different data, which would be a much worse failure than one extra model call. See bug #8 above.
- **The dedup map is per-indexing-run, not persisted across separate `build_index.py` invocations** — if you index `file_a.pdf` today and `file_b.pdf` tomorrow, and they happen to share an identical embedded image, `file_b.pdf`'s copy will still get its own vision-model call rather than reusing `file_a.pdf`'s description from the day before. This is a real, known limitation of the current dedup scope, not something the code tries to hide — cross-run dedup would need a hash lookup against the docstore itself, not just an in-memory dict scoped to one `summarize_elements()` call.

## Rebuilding the index / cleaning up

`build_index.py` with **no arguments** scans `assets/` and indexes everything supported that isn't already indexed — this is the normal way to keep the index in sync as you add files, one command regardless of how many files you've dropped in. Everything the pipeline builds is disk state under this directory — deleting it and re-running `build_index.py` is always safe and always gets you back to a known-clean state.

| To do this | Run this |
|---|---|
| Index every new file you've dropped into `assets/` (the normal, everyday case) | `uv run build_index.py` — scans the whole directory, skips anything already indexed, processes only what's new |
| Rebuild EVERYTHING from scratch (e.g. you edited a PDF in place, or changed `VISION_MODEL`/`EMBEDDING_MODEL` and want fresh summaries/vectors) | `uv run build_index.py --force` — wipes the shared index, then re-indexes every supported file currently in `assets/`, not just one |
| Re-index (or index for the first time) just ONE specific file, without touching anything else already indexed | `uv run build_index.py --file your_file.pdf` (or `--pdf`, kept as an alias) — targets a single file; if it's already indexed this is a no-op unless you also pass `--force`, which — same caveat as above — wipes the *entire* shared index, not just this file, then rebuilds from every file in `assets/` |
| Wipe the vector index and docstore, keep your files in `assets/` | `rm -rf .chroma_multimodal/` — `assets/*` is untouched; next `build_index.py` call rebuilds from scratch |
| Wipe all saved sessions (fresh start on `--chat`/`--resume`/`--list`) | `rm -rf .sessions/` — the vector index and docstore are untouched; this only affects conversation history, not what's searchable |
| Full reset — everything this pipeline has ever built, source files untouched | `rm -rf .chroma_multimodal/ .sessions/` |
| Check what's currently indexed without rebuilding anything | `uv run build_index.py` when everything in `assets/` is already indexed — prints "already indexed, skipping" for each file plus the final list of currently-indexed sources, and does nothing else |

None of this touches `assets/` — your source files are never deleted by anything in this pipeline; `.chroma_multimodal/` and `.sessions/` are the only two directories any command here writes to, and both are safe to delete any time you want a clean slate (both are gitignored, so deleting them never shows up as a change to commit).

## Where this goes next

This is [Stage 5](../../docs/stage-05-rag-retrieval.md)'s spectrum table's "agentic RAG" row, [Stage 6](../../docs/stage-06-sessions-state.md)'s session/event persistence, and multimodal ingestion over a growing document collection, combined into one pipeline. Compare against [`stage05-rag-langgraph`](../stage05-rag-langgraph/) for the same corrective-RAG logic expressed as an actual `StateGraph` instead of a function chain, and [`stage06-sessions-plain`](../stage06-sessions-plain/) for the session mechanics on their own, without retrieval layered on top.
