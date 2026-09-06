[← Back to roadmap](../../README.md)

# News/PDF Event-Storyline RAG — Implementation Plan

**Status: a design document, deliberately.** This folder has no code, and that's the point — it's the one place in this repo where an architecture is worked out *on paper* before any implementation exists, which is a skill the rest of the roadmap asserts matters but never demonstrates. Read it for the reasoning, not for something to run. If you do build it later, this is the spec to build against.

## The problem this solves

A naive "scrape news → summarize with an LLM → embed the summary → RAG" pipeline works for isolated questions ("what did this article say") but breaks down on the actual goal: **understanding an evolving storyline across many articles over time.**

Concretely, this sequence of real news about one company:

```
Aug 1  — Company announces ₹5,000 Cr capex for a new plant
Aug 5  — Company receives environmental approval
Aug 12 — Company awards the construction contract
Aug 18 — Company delays the project by 6 months
```

A correct summary of the Aug 18 article is **not** "the company announced a delay." It's:

> "This is related to the previously announced ₹5,000 Cr capex project. The project received environmental approval and a construction contract earlier this month, but the company is now delaying implementation by six months."

Plain vector-similarity RAG over raw articles doesn't reliably produce this. Two articles about the *same* project use different words each time (an announcement doesn't read like a delay notice), so embedding similarity alone is a weak signal for "these belong to the same storyline" — and it's an equally weak signal for the opposite mistake: two *different* companies' similarly-worded EV announcements can look "related" by pure text similarity when they have nothing to do with each other. Company/entity identity has to be a **hard filter**, not a fuzzy similarity contributor.

## Why not vectorless RAG

Vectorless RAG (an LLM navigating a document's table of contents instead of embeddings) doesn't fit here. Vectorless works because a *single, well-structured document* has stable navigable structure — headings, a table of contents. A stream of news articles across companies and months has no such structure: there's no table of contents spanning "everything ever published about Company X." This is a **cross-document, temporal** retrieval problem, which is exactly what vector search (used correctly — see below) is for.

## Why not "just RAG" either

Both extremes are wrong for this problem:

- **Naive RAG** (embed every article, retrieve top-k, summarize) — loses the storyline. Each summary is written in isolation; nothing tells the LLM that today's article continues a thread from three weeks ago.
- **"Send the LLM everything"** (concatenate all prior articles about this company into the prompt) — doesn't scale. At 100,000 articles and 20,000 events, no context window holds "everything," and cost/latency degrade badly well before you get there.

The actual fix, and the core design decision of this document: **separate the *article* (raw evidence) from the *event* (a structured fact extracted from it) from the *storyline* (accumulated state across many events).** Store all three. Retrieve narrowly — a new event only needs the current state of *its* storyline plus a handful of genuinely related past events, not the whole history.

## Architecture

The pipeline is five stages, in order: documents come in from news, PDFs, RSS, or the web and get fetched and parsed; a first LLM call extracts a structured event from each article — company, event type, amount, date, entities — a narrow, close-to-deterministic job; retrieval then filters by company/ticker as a hard gate and only *then* ranks by vector similarity over that company's past events (see "Why entity filtering is not optional" below); a second LLM call takes the new event, the current storyline state, and a few related past events, and produces a summary that explains what *changed*, not just what happened; and finally everything is stored — the raw article, the structured event, the contextual summary, and the storyline state, all three kept, not just the summary (see "Store three things, not one").

Two LLM calls with two different jobs, not one giant prompt doing both — extraction and reasoning are different kinds of tasks, and keeping them separate keeps each one narrow enough to grade independently.

## Data model

Four kinds of records, in Postgres (with `pgvector` for the embedding column) — not a graph database, not on day one. See "Why Postgres, not a graph database yet" below.

- **Articles** — the raw evidence: source URL, source type (news/PDF/RSS), the company it's denormalized against for fast filtering, when it was published, and the full raw text. Never deleted or edited after ingestion — this is the permanent record everything else derives from.
- **Events** — one structured fact extracted from an article. An article can produce zero, one, or several events (a long article might announce a capex project *and* report a quarterly earnings figure — two separate events). Each event carries the company (the hard filter key retrieval depends on — see below), an open-vocabulary event type, the date the event actually *happened* (not the article's publish date), amount/currency where applicable, a short plain-language summary text (the piece that gets embedded), and the full structured extraction for anything not worth its own column.
- **Event relationships** — explicit links between two events, once a reasoning step (or in Phase 1, a human/simple rule) has judged they're connected: follow-up, update, contradiction, supersession, confirmation, or explicitly unrelated. This is what makes a storyline queryable later — Phase 3 reads from these relationships rather than reconstructing them from scratch each time.
- **Storylines** (Phase 3 only, deliberately not built in Phase 1) — accumulated state for a named thread of events about one company: a title, a status, and a current-state object summarizing where things stand right now.

Phase 3 also needs two more pieces of bookkeeping, both explained in "Storyline merging, concretely" below: an audit log of every time an event's storyline assignment is made or changed (so a merge is a recorded event, not a silent overwrite), and a review queue for borderline storyline-membership decisions that shouldn't be auto-decided.

**The event summary is the only thing in this design that gets embedded.** Articles themselves are never embedded — only the structured, LLM-extracted event summary is. This is deliberate: embedding a whole raw article mixes signal (the actual fact) with noise (boilerplate, quotes, unrelated context in the same piece) — embed the *summary*, keep the *raw content* linked separately for when you need the full source.

## Why entity filtering is not optional

This is the single most important implementation detail, worth stating plainly: **retrieval for a new event must filter by company (an exact match, applied as a hard precondition) *before* vector similarity ever runs, never blend company-match into the similarity score.**

Two unrelated companies announcing similarly-worded EV plants will have close embeddings — that's expected and correct behavior for a text embedding model, and exactly why it's dangerous here. If company filtering is soft (a scoring factor rather than a hard gate), a wrong-company match can outrank a correct same-company match that happens to be worded less similarly. The fix is structural, not a tuning parameter: filter by company first, rank by embedding distance second, always in that order — a mandatory constraint, not an optional flag.

## Proposed file layout

One file per pipeline stage, so each part can be read and modified independently: a database module owning the schema and connection handling (raw SQL, no ORM — simpler to reason about than an ORM abstraction for a handful of tables and queries); an ingest package with one source module per input type (PDF filings and news/RSS each need their own fetcher — a news article shouldn't be forced through a PDF-shaped parsing path); an extraction module for LLM #1 (article → structured events, see "Extraction, concretely" below); a retrieval module for entity-filtered vector search plus grading (see "Retrieval, concretely" below); a contextual-summary module for LLM #2 (new event + retrieved past events → a summary of what changed); a pipeline module wiring the stages together; and a CLI entry point for ingesting a source or viewing a company's timeline.

Phase 3 adds a relationship-classification module and extends the contextual-summary step's output to also update a storyline's current state — deliberately not scaffolded above, since it only applies once storylines exist as a concept in storage.

## Extraction, concretely (Phase 1)

The extraction step needs a strict output schema, enforced through structured-output mode that forces the model's response to conform to a schema, not "ask nicely and hope the JSON parses."

Each extracted event carries: the company as raw-extracted text (normalization happens in a separate step, see below); an open-vocabulary event type (not a fixed enum — see "Event types" below); the date the event itself happened, distinct from the article's publish date, and nullable when the article doesn't state it; an amount and currency where applicable; any other entities mentioned beyond the primary company; a short plain-language summary (the piece that actually gets embedded); and the model's own confidence score for the extraction, stored for later use rather than acted on immediately. An article can yield zero, one, or several such events — an empty result is valid (an opinion piece with no concrete event in it).

The extraction prompt's job is narrow and close to deterministic: identify every distinct, concrete event about a company — an announcement, an approval, a delay, a financial result, a leadership change — while explicitly excluding general commentary and background context the article merely restates. It's instructed to prefer the date the event itself happened over the article's publish date when the two differ and both are stated, and to reflect uncertainty in the confidence score rather than guessing silently. Extraction runs at zero temperature — this is a data-extraction task, not a creative one.

**Company name normalization** (from the open questions below) belongs as its own small step called right after extraction, before anything is written to storage — a lookup against a simple alias table (raw variant → canonical name) that starts empty and gains entries as real variants are observed. Deliberately not solved with fuzzy string matching alone at first — an explicit, inspectable alias table is more debuggable than a similarity threshold that silently merges or splits companies.

## Retrieval, concretely (Phase 2)

This is the one part of the whole plan where getting the filter order wrong silently breaks the entire design's safety property, so it's worth being precise about, not just gesturing at: retrieval for a new event applies the company filter as a hard precondition first, then ranks by vector distance over only the records that survive that filter — never a single combined score that blends company match and semantic similarity together. Distance here means closer-is-more-similar (whichever of cosine or L2 the vector index uses), computed only against other events for the same company.

Two more constraints shape the query beyond the company filter: only *past* events (relative to the new event's date) are eligible as context — never a later, unrelated event outranking an actually relevant earlier one purely on embedding distance — and each candidate's summary text, not the raw article, is what's compared, for the same signal-over-noise reason summaries are what's embedded in the first place.

A returned candidate is not automatically trusted just because it passed the filter and ranked well: a grading step asks whether a candidate is genuinely relevant context for the new event — part of the same storyline, not just topically adjacent — before it's allowed into the contextual-summary step.

## Dependencies and local setup (Phase 1)

The dependency set is small: a Postgres driver (raw SQL, no ORM, per the file-layout note above), a schema-validation library for the extraction/relationship structured outputs, a local-model client for extraction/grading/contextual summary, a PDF-parsing library, and an RSS-parsing library if that ends up being the news-feed mechanism.

The one piece of infrastructure this plan needs beyond the model runtime: a local Postgres instance with the `pgvector` extension enabled, and the schema created against it before Phase 1's build sequence starts.

Models — extraction and grading can share one general-purpose local chat model; embeddings use a separate embedding model. No vision model is needed for Phase 1-3 as scoped (text-only extraction) — only add one if a source PDF's key facts live in a chart/table image.

## Build sequence (Phase 1, in order)

A concrete order to build and test Phase 1 in, so each step is independently verifiable before the next depends on it:

1. **Create the schema, minimally.** Only the articles and events records are needed to start — skip relationships and storylines entirely, since Phase 1 doesn't touch them. Set up a local Postgres with the `pgvector` extension enabled first.
2. **Get one real PDF (a company filing) turned into a stored article.** Verify by reading the stored raw text back and comparing it to the source — no LLM involved yet.
3. **Run extraction against that one article's raw text, in isolation.** Print the result and read it by eye against the source PDF. Do not write anything to the database at this step — verify the extraction is right first, in isolation, before trusting it inside the full pipeline.
4. **Wire extraction into storage.** Article in, extraction runs, each event's summary gets embedded, event records get written. Verify by querying the events for one company ordered by date — this is Phase 1's actual "Done when" bar.
5. **Add the second source type (news/RSS) only after steps 1-4 work end-to-end for PDFs.** Verify the same extraction/storage path handles a news article correctly without any change to the extraction step — if it needs a change, that's a sign the extraction prompt or the article-text assumption was accidentally PDF-specific.
6. **Feed 5-10 real articles about the same real company.** This is the first point real storyline behavior becomes checkable by eye: does the resulting event timeline read correctly in order, with correct dates and event types? Fix extraction prompt issues here, against real data, before writing a single line of Phase 2 code.

Phase 2 and Phase 3 each deserve their own similarly concrete build sequence once Phase 1 is running against real data — deliberately not written out yet, since Phase 2's retrieval-quality tuning depends on what Phase 1's real extracted events actually look like, not on a plan made in advance of seeing them.

## Phased build order

Building all eleven pieces of the full architecture at once means the hardest, least-certain part (storyline-membership judgment) gets designed from imagined examples instead of real data. Build in three phases; each is independently shippable and testable.

### Phase 1 — Article → Event → Postgres (build this first)

**Scope:** ingestion, extraction, storage. No relationship detection, no storyline state yet.

- A fetcher/parser per source type (PDF filings and news/RSS each need their own, separate module — do not try to force a news article through a PDF-shaped parsing path).
- LLM #1 (extraction): one article in, one or more structured events out. Keep this prompt narrow and closer to deterministic — "extract company, event type, amount, date, entities" is a data-extraction task, not a reasoning task, and should be graded/validated like one (a schema the model's output must conform to).
- Embed each event's summary text (not the raw article).
- **Done when:** you can query the events for one company, ordered by date, and get a real, correct, ordered timeline of extracted facts — no LLM call needed to read it back.

This phase alone is useful independent of everything after it — a structured, queryable event timeline per company, built from unstructured news, is a real deliverable.

### Phase 2 — Entity-filtered retrieval + contextual summary

**Scope:** given a new event, retrieve related past events (same company, vector-ranked) and have LLM #2 write a summary that references them if relevant.

- Retrieval-then-grading, in two clear steps: retrieval is "vector search over event summaries, filtered to this company"; grading is "is this candidate event actually relevant to the new one, or just topically adjacent" — a distinct check, not folded into the retrieval ranking itself.
- LLM #2 (contextual reasoning): new event + top-k graded-relevant past events → a summary that explicitly says what changed, referencing the prior event(s) by name when relevant. **Do not skip grading here** — an ungraded "just take the top-k similar events as context" step is exactly the failure mode corrective retrieval is meant to prevent, and here a wrong "related" match doesn't just produce one bad answer, it can poison every *future* summary that cites this one as context.
- Store the contextual summary alongside the event (not replacing it — see "Store three things, not one").
- **Done when:** feeding the Aug 18 delay article produces a summary that correctly references the Aug 1/5/12 events for the same company, and feeding an unrelated company's similarly-worded announcement does *not* pull in the wrong company's history.

### Phase 3 — Storyline state (build only after Phase 2 is proven on real data)

**Scope:** event relationships populated by an explicit relationship-classification step, and a storylines record maintaining running state per thread, so a new event updates existing state instead of the summary step re-deriving everything from scratch each time.

- A dedicated classification call: "does this new event belong to an existing storyline, or start a new one?" with a fixed relationship vocabulary (follow-up, update, contradiction, supersession, confirmation, unrelated) — same structured-output discipline as Phase 1's extraction, not a free-text judgment.
- A storyline's current-state object gets updated, not re-summarized from the full history, on each new related event — this is what keeps LLM #2's context small even as the underlying event count grows into the thousands (feed it *current state* + *this new event* + a handful of the most relevant past events, never the full history).
- **Explicitly deferred to this phase, not before:** this is the part of the architecture with the most design risk (storyline-membership boundaries are genuinely ambiguous — does a "quarterly earnings beat" belong to the same storyline as a "capex delay" for the same company, or are they separate threads?). Building this after Phase 1/2 have produced real extracted events means the classification prompt and the relationship vocabulary get designed against real ambiguous cases, not imagined ones.

### Storyline merging, concretely — three bands, not one threshold

"Match on company + project name, fall back to vector similarity" (Phase 2's shape) is not precise enough on its own for Phase 3's harder question: **does this new event belong to an EXISTING storyline, or start a new one?** A single similarity cutoff (e.g. "merge if similarity > 0.8") silently produces two failure modes at once — false merges just above the line, false splits just below it — with no way to tell which one happened without manually auditing the data later. The fix is **three explicit bands**, not one number:

- **High confidence (similarity above roughly 0.90):** merge automatically. The new event is attached to the existing storyline without a human in the loop, and the decision is recorded as an automatic high-confidence merge.
- **Low confidence (similarity below roughly 0.60):** treat as clearly unrelated and start a new storyline automatically. No human needed here either — the candidate is confidently a non-match.
- **The borderline band in between:** neither confident merge nor confident new-storyline. Rather than guessing, the event is queued for a human to resolve, and in the meantime gets a provisional storyline of its own so nothing is left unassigned while waiting on a decision.

Both thresholds (roughly 0.90 and 0.60) are **placeholders to replace with real numbers once Phase 3 has actual similarity-score data to look at** — not values to trust from a plan written before seeing real embeddings. The band width matters more than the exact numbers: too narrow and almost everything lands in manual review, defeating the purpose of automation; too wide and the auto-merge/auto-new bands make silent mistakes just as a single threshold would. Tune both thresholds against a labeled sample of real borderline cases — a small golden dataset built specifically for this decision — before trusting either in production.

**Resolving a queued review** is a distinct, explicit step, not left implicit: a human decision (merge or reject) either attaches the event to the candidate storyline it was queued against, recorded as a human-approved merge, or — if rejected — leaves the event's provisional storyline as its permanent one. Whichever mechanism surfaces the queue to a reviewer (a CLI prompt to start, a real UI later) is free to change without touching how a resolution is recorded.

Every merge — automatic or human-approved — is logged, including the very first assignment an event ever gets (with no prior storyline to record). This is what makes an event's storyline assignment safe to treat as mutable: the log is the audit trail answering "why does this event belong to this storyline, and did that ever change," which a bare silent overwrite would lose.

**Open question this doesn't resolve yet:** what gets embedded to represent a *storyline* (as opposed to a single event) for the candidate-comparison step above — the most recent event's summary, a text rendering of the storyline's current state re-embedded on the fly, or a dedicated embedding column maintained alongside that state? Leaning toward the third option, since it represents "what this storyline is about *now*," not just its most recent event — but this needs deciding with real data, not in this planning doc.

### Contradiction handling — when a new event conflicts with storyline state

The relationship vocabulary already includes a `CONTRADICTS` type, but a classification label alone doesn't specify *what happens to a storyline's current state* when one fires — and this is exactly the case most storyline-tracking designs quietly get wrong: assuming the reasoning step (LLM #2 / the contextual-summary step) will "just handle it" by writing a sensible new summary. It might, most of the time — but the current-state object feeds future retrieval and summaries, it isn't prose a human reads once and discards. Quietly overwriting a status field from "delayed" to "on schedule" with no record of the contradiction is a state transition that should be explicit and auditable, the same way the storyline-merge log makes merges auditable instead of a silent overwrite.

The fix: a contradiction is handled as its **own branch** in the state-update step, not folded into the generic "update state" path every other relationship type uses. When a new event contradicts the storyline's current state, the conflicting fields are identified, the contradiction itself is recorded, and the new state is produced by an explicit reconciliation step — never the default "just summarize everything again" path. Every other relationship type (follow-up, update, supersession, confirmation) extends state additively, with no conflict to reconcile, so those go through the ordinary update path directly.

**The reconciliation step's simplest honest v1 is a rule, not an LLM call**: the most recent event's claim wins, but the contradiction and the overridden prior claim are both preserved in the state's history, not deleted. A status field becomes "on schedule," but the state's history gains an entry noting it was "delayed" as of the earlier date, with a pointer to which event caused each change. This is deliberately conservative — it never requires trusting an LLM to correctly adjudicate a real contradiction (a genuinely hard reasoning task) before Phase 3 has any real contradictions to test that judgment against. Escalate to an LLM-based reconciliation step later, once "most recent wins" has been observed to be insufficient on real data — for instance a contradiction that isn't a simple supersession, where two events both claim to be current and can't both be right.

### Why store three things, not one

Every event should retain **all three** representations, never just the final summary: the raw article text, the structured event extracted from it, and the contextual summary written about it. Summaries are lossy by construction. Six months from now, a question might need a specific figure or clause that didn't make it into the summary written at the time. If only the summary was kept, that information is permanently gone — re-deriving it means re-fetching a source that may no longer be available. Storing all three costs disk space, which is cheap; it does not cost re-engineering later, which is not.

### Why store negative relationships too

Event relationships should record explicitly-unrelated judgments, not just positive links. A candidate that vector search surfaced and the relationship step explicitly rejected is worth keeping a record of — it's evidence the classification step considered and ruled out this pairing, which matters for debugging ("why didn't this get linked?") and for not re-asking the same question if the same candidate pair comes up again.

### Why Postgres, not a graph database yet

Event relationships form a straightforward edge list — one event, a related event, and a relationship type — and Postgres handles that natively with a normal indexed join; a graph database earns its complexity once traversal patterns get deep (e.g. "find all events three hops away that share an entity but not a direct link"), which Phase 1–3 above do not need. Reaching for a graph database before the relationship model is proven on real data is optimizing a part of the system that hasn't been validated yet. Revisit this decision after Phase 3 is running against real data, not before.

### Event types — keep the vocabulary open, not a fixed enum

An event's type is a free-text field with example values (capex announcement, regulatory approval, project delay, and so on) suggested in Phase 1's extraction prompt, not a hard-coded enum in the schema. A fixed enum decided before seeing real data will be wrong — real news produces event types you won't have anticipated (a leadership change, a lawsuit, a credit rating change), and a schema migration to add an enum value is friction a free-text field with a documented-but-not-enforced vocabulary avoids. Tighten this into a real enum later, once Phase 1 has run against enough real articles to know the actual distribution of event types.

## General principles this design follows

These aren't specific to this project — they're patterns worth applying to any agentic RAG system with a similar shape, independent of which codebase they're built in:

- **Extract structured facts with one model call, reason over them with a separate model call.** Don't ask a single prompt to both pull out the concrete data (who, what, when, how much) and write the narrative interpretation of it. Extraction is a data task with a right answer; reasoning over accumulated context is a judgment task. Splitting them keeps each prompt narrow enough to grade, and keeps a bug in one from silently corrupting the other.
- **A hard identity filter (company, user, tenant, whatever the entity boundary is) always runs before similarity ranking, never blended into a similarity score.** Semantic similarity is a weak, gameable signal for "these belong together" whenever two unrelated things can be worded almost identically. Anywhere that risk exists, the identity check has to be a precondition on the candidate set, not a factor added into the ranking.
- **Never trust a retrieved candidate just because it ranked well.** A cheap first-pass retrieval step (vector search, keyword search, whatever) should always be followed by an explicit relevance check before the candidate is allowed to influence an answer or a piece of persisted state. Skipping this is exactly how one bad match quietly poisons everything downstream that later cites it.
- **Validate any model output that becomes structured data against an explicit schema**, rather than parsing free text and hoping. This applies wherever an LLM's output is about to be written to storage or branched on — extraction results, classification labels, judgments — not just user-facing answers.
- **Keep raw evidence, the structured fact extracted from it, and any narrative summary built on top of it as three separate, permanently retained things — never collapse down to just the summary.** Summaries are lossy by construction; re-deriving lost detail later usually isn't possible once the source is gone. Storage is cheap; re-engineering after the fact is not.
- **A single similarity threshold for a consequential, hard-to-reverse decision (merge these two things, or don't) produces two invisible failure modes at once — false positives just above the line, false negatives just below it — with no way to tell which happened without an audit.** Replace one threshold with two, creating a confident-yes band, a confident-no band, and a borderline band that gets queued for a human instead of auto-decided.
- **Any decision that reassigns a record from one grouping to another after its first assignment needs an audit trail, not a silent overwrite.** If a system will ever move an item from one bucket to another (a merge, a re-categorization, a corrected label), log every such change — including the very first assignment — so "why does this belong here now, and did that ever change" stays answerable.
- **A contradiction is not the same event type as an update, and shouldn't be handled by the same code path.** When new information conflicts with existing stored state rather than simply extending it, route it through a distinct, explicit reconciliation step that records the conflict and preserves the overridden prior claim in history. The simplest honest first version of that step is a plain rule ("most recent wins, but nothing is deleted"), not an LLM call — reserve the harder judgment call for once real contradictions exist to test it against.
- **Keep any open-ended categorical field (event types, relationship types, statuses) as free text with a documented-but-unenforced vocabulary until real data reveals the actual distribution.** A fixed enum decided before seeing real data is reliably wrong in ways that cost a schema migration to fix; a text field with example values costs nothing to extend.
- **Don't reach for a more complex storage model (a graph database, a specialized index) until the simpler one has been proven insufficient on real data.** A straightforward relational edge list handles most relationship-tracking needs; the complexity of graph traversal is only worth paying for once query patterns that actually need it show up.

## Open questions to resolve before or during Phase 1

Written down now so they aren't forgotten, not because they're answered here:

- **Company name normalization.** News sources spell company names inconsistently ("Tata Motors" vs. "Tata Motors Ltd" vs. "TML"). The hard company filter this whole design depends on is only as good as a normalization/aliasing step that hasn't been designed yet — this is worth solving explicitly in Phase 1, not discovered as a bug later.
- **What counts as "the same event" from two different sources.** If two different news outlets both report the same capex announcement, is that one event with two articles linked, or two separate events? (Leaning toward: two separate events, since Phase 3's relationship step can mark them as confirming each other — but this needs a real decision, not an implicit default.)
- **Extraction confidence / partial extraction.** What happens when LLM #1 can't confidently extract a clean amount or event date from a vaguely-worded article? The raw structured extraction can hold a partial/low-confidence result, but the downstream retrieval/summary steps need to know to treat it differently than a clean extraction — not designed yet.

**Resolved since the sections above were first written** (kept here as a changelog note, not re-listed as open): storyline-merge threshold ambiguity and contradiction-handling were both originally open questions in this plan. Both are now designed in detail in "Storyline merging, concretely" and "Contradiction handling" above, including the one open question those sections themselves introduce (what exactly gets embedded to represent a storyline — see the end of "Storyline merging, concretely").

## Open question introduced by Phase 3's design above

- **What gets embedded to represent a storyline, for the candidate-storyline comparison.** Candidates: the most recent event's summary embedding, a text rendering of the current state re-embedded on every update, or a dedicated state embedding maintained alongside that state. See "Storyline merging, concretely" above for the current leaning (the third option) and why — this needs deciding with real Phase 3 data, not in this planning doc.

## Sources

| Type | Resource |
|------|----------|
| Tool | [pgvector](https://github.com/pgvector/pgvector) — the Postgres extension this design's vector column relies on |

## Done when

**Phase 1:** a real news/PDF source produces correctly-structured event records queryable as a clean per-company timeline, with no LLM call needed to read the timeline back.

**Phase 2:** a new article about an ongoing storyline produces a summary that correctly references the right prior events for *that* company, and a similarly-worded article about an unrelated company does not cross-contaminate.

**Phase 3:** a new event correctly updates an existing storyline's state (rather than the system re-deriving the whole history from scratch), a genuinely new storyline gets created rather than incorrectly merged into an existing one, a borderline case lands in the review queue instead of being silently auto-decided, and a contradicting event (e.g. "delayed" → "resumed on schedule") produces a current state that preserves the prior claim in history rather than silently overwriting it.

---
[← Back to roadmap](../../README.md)
