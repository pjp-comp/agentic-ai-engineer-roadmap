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

## How it works, step by step — following one real question through the pipeline

Take the question `"How much did battery storage cost decline?"` and trace exactly what happens, function by function:

**Indexing time (`build_index.py`, already done before you ever ask a question):**

1. `ingest.parse_pdf(sample_complex.pdf)` opens the PDF with pymupdf and walks every page. For each page it: (a) calls `page.find_tables()` to find real tables, extracting each as structured rows/columns; (b) calls `page.get_images(full=True)` to find embedded images, and for each one large enough to matter, captures the raw bytes **plus a narrow window of nearby text** (see "Caption grounding" below); (c) extracts every remaining text block via `page.get_text("blocks")`, explicitly excluding any block whose bounding box overlaps a detected table (so table rows never get double-counted as garbled prose — see bug #1 below). Every piece becomes an `Element` with a `kind` (`text`/`table`/`image`), a `page`, a `source` (the PDF's filename), and a unique `element_id` like `sample_complex.pdf::image-15`.
2. `index.summarize_elements()` turns each `Element` into something embeddable. Text elements pass through untouched. Table elements go to `llama3.1:8b` with the prompt "summarize what this table shows, mention headers and notable values." Image elements go to `llava:7b` (the vision model) with the image bytes AND the nearby text, asking for a 2-3 sentence description.
3. `index.build_multi_vector_index()` embeds every summary (not the raw content) with `nomic-embed-text` into the shared Chroma collection at `.chroma_multimodal/`, and writes the **raw** content (the full table grid, or the image's generated description) into `docstore.json`, keyed by `element_id`.

**Query time (`agent.py`, this happens on every question):**

4. `retrieve()` embeds your question with the same `nomic-embed-text` model, and asks Chroma for the 4 nearest summary-vectors. For this question, the summary of `sample_complex.pdf::text-19` (a paragraph stating "Storage cost declined from $280/kWh in 2019 to $98/kWh in 2024...") lands close to the question's vector, because both are about the same concept even though the wording differs.
5. `grade()` takes each of those 4 candidates and asks `llama3.1:8b`, one at a time: "is this actually useful for answering the question, not just topically adjacent?" Only candidates that pass keep going — and *only for those*, `grade()` looks up their `raw` content in the docstore and attaches it to the result. (Candidates that fail grading never have their raw content fetched at all — no wasted work.)
6. If nothing passed grading, `rewrite_query()` asks the model to rephrase the question and `retrieve()`+`grade()` run again once (`MAX_RETRIES = 1`). If still nothing, `refuse()` returns a plain "I don't have enough information" — no LLM call, because there's nothing to reason over.
7. If something passed, `generate()` builds the final prompt from the **raw** content of every graded-relevant element (not the summaries that were searched), asks `llama3.1:8b` to answer using only that context, and instructs it to cite `(source, page)`.
8. The whole exchange — your question, the answer, both as `Event`s — gets written into a `Session` (`session.py`) and flushed to disk via `PersistentSessionService.save()`, so `--history`/`--chat` can pick it back up later.

The key thing to notice: steps 4-5 (retrieve, grade) operate on **summaries** — short, question-shaped text. Step 7 (generate) operates on **raw content** — the actual table or the image's full description. These are deliberately different pieces of text for the same element; see the next section for why.

## Caption grounding — how images get described accurately (and where it still fails)

A vision model summarizing a chart from pixels alone is prone to hallucination — inventing numbers, category names, or trends that aren't actually in the image. This example mitigates that (but does not fully solve it — see "Known limitations" below) with **caption grounding**: every image `Element` carries a `caption` field, populated in `ingest.py` by taking the image's own bounding box (`page.get_image_rects()`) and collecting only the text blocks within a tight vertical window (`CAPTION_WINDOW_PT = 80` points, roughly one paragraph) above and below it — not the whole page's text.

That distinction matters concretely: an early version of this fix used the *entire page's* text as context. On `sample_complex.pdf` page 4, which has both a workforce chart AND an unrelated investment table below it, that caused `llava:7b` to pull numbers from the investment table (`231`, `112`, `84`...) and misattribute them as the chart's own category values — a worse failure than having no caption at all. Narrowing to a tight window around the image's own position on the page fixed that specific failure mode.

`index.py`'s `_summarize_image()` then passes this narrow caption to the vision model alongside the image bytes, with an explicit instruction: *if the text states a specific number, trust the text over your own reading of the image* — because a real PDF's prose usually states a chart's headline figure in words right next to it ("Installation added 680,000 jobs..."), and reading that sentence is far more reliable than reading a bar's height from pixels.

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

A more instructive refusal — a question whose answer genuinely exists in the document, but only inside a chart image the vision model can't read reliably:

```bash
uv run agent.py "How many jobs added in 2024 in manufacturing sector?"
```

This retries once with a rewritten query, still finds nothing it's confident about, and refuses — the correct, honest behavior for information the pipeline can't reliably extract, rather than confidently guessing wrong. See "Known limitations" below for the full investigation into why.

## Real bugs this surfaced, and the actual fixes

Four genuine failures came up during development, all fixed (or, for #4, honestly documented rather than papered over) in the code:

**1. Table text was being extracted twice.** The first version of `ingest.py`'s text/table separation used a length heuristic (drop any text "paragraph" shorter than 40 characters, on the theory that table rows are short). This silently failed: pymupdf's plain text extraction returns each table row as its own line, and real rows like `"Northern Europe   58.2%   64.7%   +6.5 pp"` are *longer* than 40 characters, so they passed the filter and got embedded a second time as garbled text elements alongside the properly-extracted table element. The fix: extract text as **blocks with real bounding boxes**, and drop any block whose bbox geometrically intersects a detected table's bbox — geometry, not a length guess, is what correctly separates "this text is the table" from "this text is a normal paragraph." See `ingest.py`'s `parse_pdf()`.

**2. The grader was too lenient on bare section headings.** Asking "What is the capital of France?" initially returned a confused non-answer instead of a clean refusal, because a 12-character summary ("1. Overview") got graded "relevant" — with almost no content to reason over, the small model tended to guess yes rather than commit to no. Two fixes: `grade()` now skips anything under 30 characters outright (too short to plausibly answer anything on its own), and the grading prompt was tightened from "discusses the same topic" to "contains information that's ACTUALLY useful... not just a vaguely related topic or section title." As a second layer of defense, `generate()`'s prompt was also hardened to refuse plainly rather than comment on the context itself if what it's given genuinely doesn't answer the question — a system shouldn't rely on grading catching every weak case perfectly.

**3. First version indexed each PDF into its own isolated directory.** Adding a second PDF meant a second, disconnected `.chroma_multimodal_<name>/` store, and querying required knowing in advance which one to open — the opposite of "just add more PDFs." Fixed by moving to one shared collection with a `source` field on every element (see "One shared index, not one per PDF" above) — the same fix that makes `--source` filtering possible at all.

**4. Asking about a specific bar's value on a 5-category workforce chart returned a confused refusal instead of an answer** ("how many jobs added in 2024 in manufacturing sector?") — traced all the way down to `llava:7b` inventing category names and values wholesale when describing that chart. See "Known limitations" below for the full investigation and why this is a genuine model-capability ceiling, not a bug fixable by more prompt tuning.

## Known limitations — the vision model can't reliably read a multi-bar chart

This is worth stating plainly rather than only discovering by hitting it: **`llava:7b` (a 7B-parameter local vision model) is not reliable at reading discrete values off a chart with several categories.** This was found while investigating why "how many jobs added in 2024 in manufacturing sector?" against `sample_complex.pdf`'s workforce chart (5 categories: Manufacturing, Installation, R&D, Grid Operations, Policy/Admin) returned a refusal instead of an answer.

The investigation, in order:

1. The chart image itself is completely legible to a human — a simple horizontal bar chart, 5 bars, clearly labeled.
2. `llava:7b`'s summary of the full chart invented a multi-year trend narrative on what is actually single-year data, wrong/merged category names, and a fabricated "200,000 jobs" total figure that appears nowhere in the real data. This was the caption-grounding fix's original motivation.
3. Even asked the simplest possible isolated question — *"list only the category names on the vertical axis, top to bottom, nothing else"* — `llava:7b` returned a 16-item list mixing 4-5 real category names with a dozen fabricated ones. This ruled out "the summarization prompt is too complex" as the explanation.
4. Tightening the summarization prompt to explicitly demand every category and value made fabrication measurably **worse**, not better — the model invented more content to satisfy the completeness requirement rather than admitting uncertainty about categories it couldn't actually read.

What genuinely helps (kept in the final code): grounding on nearby caption text lets the model correctly report a number when the surrounding prose *states it in words* (e.g. "Installation added 680,000 jobs" is now reliably picked up, because that's reading a sentence, not a bar's height). What doesn't help: asking the model to read a specific bar's height directly, when that number exists **only inside the image** and nowhere in the surrounding text — Manufacturing's 420 (thousand jobs) is exactly this case, since the caption text only ever mentions Manufacturing in passing, comparing it unfavorably to Installation, never stating its own value.

**Practical consequence:** a question whose answer is a *minor* bar's specific value, stated only inside a chart image, may correctly refuse rather than answer — which is the honest outcome (refusing is correct behavior for information the pipeline genuinely can't extract), but is a real gap versus what a human reading the same PDF could tell you at a glance. The actual fix isn't more prompt engineering against this model's ceiling — it's a stronger vision model (`qwen2.5vl:7b` or `llama3.2-vision:11b` are worth trying as drop-in replacements for `VISION_MODEL` in `index.py`, both reportedly stronger at reading structured chart/table content than `llava`), or a table-extraction-style approach for charts specifically (e.g. asking the model for OCR'd axis tick values rather than a holistic description). Neither is implemented here — this is a documented, known boundary of what this example's pipeline can do, not a silently swallowed failure.

## What to look at closely

- **`grade()` grades the SUMMARY, not the raw content** — cheap and fast, since pulling full raw content (a whole table, or re-describing an image) for every one of the k candidates, most of which won't pass grading anyway, would be wasted work. Only elements that pass get their raw content attached, in `grade()`'s final step.
- **`generate()` explicitly uses `c["raw"]`, never `c["summary"]`** — this is the entire point of the multi-vector split made concrete in code: the vector that got matched during search is deliberately a different string from what gets shown to the answer-writing model.
- **`retrieve()`'s `source_filter` becomes a Chroma metadata `filter={"source": ...}`** — this is a real, structural narrowing of the search space, not a post-hoc filter applied to results after the fact; Chroma only considers vectors matching that filter in the first place.
- **Small images get filtered out during ingestion** (`ingest.py`, `len(image_bytes) < 5000`) — page decorations, bullet icons, and logos aren't worth a vision-model call and would only add retrieval noise. A genuinely important but small diagram could theoretically get skipped by this threshold, worth knowing if you point this at your own PDF.
- **Caption text is windowed by vertical position, not by paragraph count** (`ingest.py`'s `CAPTION_WINDOW_PT = 80`) — a fixed distance in PDF points rather than "the paragraph before and after," which is simpler but means a very short paragraph right next to the image and a very long one just outside the window get treated inconsistently. See "Known limitations" above for why this exists and what it does and doesn't fix.
- **`--chat` and `--resume` share `_answer_turn()` and `_load_session()`** rather than duplicating turn logic — resuming a session isn't a special code path, it's the same turn function applied to whichever `Session` object got loaded, same principle as [`stage06-sessions-plain`](../stage06-sessions-plain/)'s `run_session()`.

## Where this goes next

This is [Stage 5](../../docs/stage-05-rag-retrieval.md)'s spectrum table's "agentic RAG" row, [Stage 6](../../docs/stage-06-sessions-state.md)'s session/event persistence, and multimodal ingestion over a growing document collection, combined into one pipeline. Compare against [`stage05-rag-langgraph`](../stage05-rag-langgraph/) for the same corrective-RAG logic expressed as an actual `StateGraph` instead of a function chain, and [`stage06-sessions-plain`](../stage06-sessions-plain/) for the session mechanics on their own, without retrieval layered on top. To add your own PDF: drop it in this directory and run `uv run build_index.py --pdf your_file.pdf`.
