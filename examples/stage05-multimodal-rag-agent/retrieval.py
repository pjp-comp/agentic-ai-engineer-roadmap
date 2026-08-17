"""
RETRIEVAL -- the search half of corrective RAG: retrieve candidates
(vector search, optionally fused with keyword search -- see "Hybrid
retrieval" below), then GRADE each one before trusting it. Isolated from
generation.py so "did we find the right thing" and "did we answer it
correctly" are two separately-testable concerns.

The corrective-RAG shape, same as stage05-rag-langgraph:

  retrieve -> grade -> (relevant found? -> generate)
                     -> (nothing relevant, haven't retried? -> rewrite_query -> retrieve again)
                     -> (nothing relevant, already retried? -> refuse)

retrieve() searches over SUMMARIES (see vision.py) -- the vector matched
against a question is a generated description, not raw table/image
content. grade() then asks a SEPARATE judgment: is this candidate
actually useful for answering the question, not just topically close?
Only candidates that pass grading get their RAW content looked up from
the docstore -- see generation.py for why raw content, not the summary
that was searched, is what the final answer gets built from.

Hybrid retrieval -- vector search + keyword search, fused:

  hybrid_retrieve() runs BOTH vector_store's similarity_search() and
  keyword_search.keyword_search() (BM25, see that file's module
  docstring for why exact-term matching catches things pure vector
  search can miss -- region names, exact figures, acronyms) against the
  SAME query, then merges the two ranked lists with Reciprocal Rank
  Fusion (see reciprocal_rank_fusion() below). This is the "Advanced/
  Hybrid" row of docs/stage-05-rag-retrieval.md's RAG spectrum table,
  built for real: pure semantic search regularly loses to a hybrid
  keyword+vector approach on anything with exact-match terms.

  retrieve() (vector-only) is kept as a separate, still-callable
  function -- hybrid isn't presented as strictly replacing vector-only
  search, it's an ADDITIONAL retrieval mode you opt into (see agent.py's
  --hybrid flag), so the two can be compared directly on the same query.
"""

import sys

import ollama
from langchain_chroma import Chroma

from keyword_search import keyword_search
from vision import LOCAL_MODEL

RETRIEVE_K = 4
MAX_RETRIES = 1
RRF_K = 60   # the standard RRF damping constant -- see reciprocal_rank_fusion()'s docstring


def retrieve(query: str, vector_store: Chroma, source_filter: str | None = None) -> list[dict]:
    # source_filter=None searches across EVERY indexed PDF at once (the
    # default -- this is the shared-index behavior); passing a filename
    # narrows the search to just that document's elements via Chroma's
    # metadata filter, using the "source" tag every element carries.
    where = {"source": source_filter} if source_filter else None
    results = vector_store.similarity_search(query, k=RETRIEVE_K, filter=where)
    candidates = [
        {"element_id": r.metadata["element_id"], "kind": r.metadata["kind"],
         "page": r.metadata["page"], "source": r.metadata["source"], "summary": r.page_content}
        for r in results
    ]
    kinds = ", ".join(f"{c['kind']}({c['source']} p{c['page']})" for c in candidates)
    print(f"  [retrieve] query={query!r} -> {len(candidates)} candidate(s): {kinds}", file=sys.stderr)
    return candidates


def reciprocal_rank_fusion(ranked_lists: list[list[dict]], k: int = RRF_K) -> list[dict]:
    """Merges multiple ranked candidate lists into one, using Reciprocal
    Rank Fusion: RRF_score(doc) = sum over every list it appears in of
    1 / (k + rank_in_that_list). This is the standard technique
    Elasticsearch, Azure AI Search, and Weaviate's hybrid mode all use
    for exactly this problem.

    WHY RANK, not raw score: a Chroma cosine-similarity score (roughly
    0-1) and a BM25 score (unbounded, depends on corpus size and term
    rarity) are not on comparable scales -- there's no principled way to
    average "0.82 similarity" with "14.3 BM25 score" directly. Rank
    position sidesteps this entirely: "this was the vector search's
    #1 result" and "this was the keyword search's #1 result" are
    directly comparable regardless of what the underlying scores meant.

    k=60 is RRF's standard damping constant (from the original paper,
    Cormack et al. 2009, and reused as the default across every major
    implementation) -- it flattens the difference between adjacent
    ranks so a document doesn't need to be #1 in both lists to win;
    being consistently well-ranked across both signals is what RRF
    rewards, which is exactly the "this is relevant by two independent
    measures" signal hybrid retrieval is trying to capture.
    """
    scores: dict[str, float] = {}
    candidates_by_id: dict[str, dict] = {}

    for ranked_list in ranked_lists:
        for rank, candidate in enumerate(ranked_list, start=1):
            element_id = candidate["element_id"]
            scores[element_id] = scores.get(element_id, 0.0) + 1.0 / (k + rank)
            candidates_by_id[element_id] = candidate   # last write wins -- identical across lists anyway, same docstore

    fused_ids = sorted(scores, key=lambda eid: scores[eid], reverse=True)
    return [candidates_by_id[eid] for eid in fused_ids]


def hybrid_retrieve(query: str, vector_store: Chroma, docstore: dict, source_filter: str | None = None) -> list[dict]:
    """The hybrid retrieval mode: vector search (semantic, paraphrase-
    tolerant) and keyword search (BM25, exact-term-precise) run
    independently against the same query, then get fused by rank
    position rather than raw score. Returns a single ranked candidate
    list in the SAME shape retrieve() returns, so grade() downstream
    doesn't need to know or care which retrieval mode produced it.
    """
    vector_results = retrieve(query, vector_store, source_filter)
    bm25_results = keyword_search(query, docstore, k=RETRIEVE_K, source_filter=source_filter)
    print(f"  [keyword] query={query!r} -> {len(bm25_results)} candidate(s): "
          f"{', '.join(c['element_id'] for c in bm25_results)}", file=sys.stderr)

    fused = reciprocal_rank_fusion([vector_results, bm25_results])[:RETRIEVE_K]
    print(f"  [hybrid] fused -> {len(fused)} candidate(s): {', '.join(c['element_id'] for c in fused)}", file=sys.stderr)
    return fused


def grade(candidates: list[dict], original_query: str, docstore: dict) -> list[dict]:
    # Grade against the SUMMARY (cheap, no need to pull raw content for
    # elements that won't pass anyway) but against the ORIGINAL question,
    # same corrective-RAG discipline as stage05-rag-langgraph.
    graded = []
    for c in candidates:
        # A real bug this surfaced: bare section headings ("1. Overview",
        # ~12 chars) are too short/generic for the grader to judge
        # reliably -- with almost no content to reason over, it tends to
        # guess "yes" rather than commit to "no." Skip grading (and thus
        # including) anything too short to plausibly ANSWER a question on
        # its own, rather than trusting the model to catch its own weak
        # input every time.
        if len(c["summary"].strip()) < 30:
            print(f"  [grade] {c['element_id']} ({c['kind']}, p{c['page']}): skipped (summary too short to be useful)", file=sys.stderr)
            continue

        verdict = ollama.chat(
            model=LOCAL_MODEL,
            messages=[{
                "role": "user",
                "content": (
                    f"Retrieved content summary ({c['kind']}, page {c['page']}):\n{c['summary']}\n\n"
                    f"Question: {original_query}\n\n"
                    "Does this summary contain information that is ACTUALLY useful for answering "
                    "the question -- not just a vaguely related topic or section title? "
                    "Answer no if it's just a heading, a generic intro sentence, or off-topic. "
                    "Reply with exactly one word: yes or no."
                ),
            }],
            think=False,
            # temperature=0: a real bug this caught -- without it, grading
            # the SAME summary against the SAME question flipped between
            # "relevant" and "not relevant" run to run, which meant a
            # correct answer (using the table) and a fabricated one
            # (guessing from an unrelated text block once the table got
            # excluded) were both reachable from an identical query,
            # nondeterministically. Grading is a binary judgment call, not
            # creative writing -- it should give the same verdict every
            # time for the same input.
            options={"temperature": 0},
        )
        is_relevant = "yes" in verdict.message.content.strip().lower()
        print(f"  [grade] {c['element_id']} ({c['kind']}, p{c['page']}): {'relevant' if is_relevant else 'not relevant'}", file=sys.stderr)
        if is_relevant:
            # Attach RAW content now, only for elements that passed --
            # this is the multi-vector technique's second half: what gets
            # handed downstream is the original content, not the summary
            # that was searched.
            record = docstore.get(c["element_id"], {})
            graded.append({**c, "raw": record.get("raw", c["summary"])})
    return graded


def route_after_grading(graded_relevant: list[dict], retries: int) -> str:
    if graded_relevant:
        return "generate"
    if retries < MAX_RETRIES:
        return "rewrite_query"
    return "refuse"


def rewrite_query(original_query: str) -> str:
    rewritten = ollama.chat(
        model=LOCAL_MODEL,
        messages=[{
            "role": "user",
            "content": (
                f"Original question: {original_query}\n\n"
                "Rewrite this as a different, more specific search query that might match "
                "a passage, table, or chart in a report on renewable energy adoption. "
                "Reply with ONLY the rewritten query, nothing else."
            ),
        }],
        think=False,
        options={"temperature": 0},
    )
    new_query = rewritten.message.content.strip()
    print(f"  [rewrite] {original_query!r} -> {new_query!r}", file=sys.stderr)
    return new_query
