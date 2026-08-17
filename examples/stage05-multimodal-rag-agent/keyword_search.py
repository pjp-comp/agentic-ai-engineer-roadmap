"""
KEYWORD SEARCH -- the second half of hybrid RAG, alongside vector_store.py's
semantic search. Implements BM25 (the ranking algorithm behind
Elasticsearch/OpenSearch's default full-text scoring) entirely in Python,
over the same element SUMMARIES vector search already embeds -- no
external search service, no new index to keep in sync, just a different
way of ranking the same underlying text.

Why hybrid at all -- what vector search alone actually misses:

  Vector search (vector_store.py) finds text that means something similar
  to the question, which is exactly right for paraphrase-tolerant
  retrieval ("what was East Asia's change" matching a summary phrased
  completely differently). But embeddings are lossy for EXACT tokens --
  a model number, a ticker symbol, an acronym, a precise figure like
  "$98/kWh," or a region name spelled a specific way can sit at merely
  "pretty close" in vector space to a paraphrase that doesn't actually
  contain it, while a chunk that literally contains the exact string can
  rank behind it. This is a well-documented, real failure mode of
  embedding-only RAG, not a hypothetical one -- see docs/stage-05-rag-retrieval.md's
  "hybrid search" row in the RAG spectrum table for the same point made
  in the roadmap docs this repo teaches from.

  BM25 is the complementary tool: it scores a document by how often the
  query's exact TERMS appear in it (with diminishing returns for very
  common words, and length normalization so a short exact match doesn't
  lose to a long document that mentions the term once). It has NO notion
  of meaning or paraphrase -- ask it "what was East Asia's change" and
  it will score on the literal words "east," "asia," "change," missing a
  summary that says "the region's year-over-year gain" entirely. Neither
  retrieval mode is sufficient alone; that's the actual argument for
  fusing them (see retrieval.py's reciprocal_rank_fusion()), not "more
  retrieval methods are always better."

Rebuilt fresh from the docstore on every query, not persisted separately --
BM25 over a few dozen-to-hundred summaries is fast enough (milliseconds)
that a second on-disk index to keep in sync with vector_store.py's
Chroma collection would be complexity without a real performance payoff
at this example's scale. A production system with thousands of documents
would persist a real BM25 index (e.g. via Elasticsearch or a library like
bm25s) instead of rebuilding per query.
"""

import re

from rank_bm25 import BM25Okapi

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    """Lowercase, alphanumeric-only tokens -- BM25 scores on exact token
    overlap, so consistent tokenization between the indexed corpus and
    the query matters more than a sophisticated tokenizer here. This is
    intentionally simple: no stemming, no stopword removal -- a keyword
    match on "adoption" vs. "adopted" won't hit, which is a real, known
    limitation of naive BM25 tokenization worth knowing rather than
    hiding behind a more elaborate pipeline this example doesn't need.
    """
    return _TOKEN_PATTERN.findall(text.lower())


def build_bm25_index(docstore: dict, source_filter: str | None = None) -> tuple[BM25Okapi, list[str]]:
    """Builds a BM25 index over every element's SUMMARY in the docstore --
    same text vector search embeds (see vision.py), so keyword and
    semantic search are ranking the SAME underlying corpus, just by two
    different signals. Returns (bm25_index, element_ids) -- element_ids
    is parallel to the index's internal document order, needed to map a
    BM25 score back to which element it came from.
    """
    if source_filter:
        items = [(eid, rec) for eid, rec in docstore.items() if rec.get("source") == source_filter]
    else:
        items = list(docstore.items())

    element_ids = [eid for eid, _ in items]
    tokenized_corpus = [_tokenize(rec["summary"]) for _, rec in items]
    return BM25Okapi(tokenized_corpus), element_ids


def keyword_search(query: str, docstore: dict, k: int, source_filter: str | None = None) -> list[dict]:
    """Returns the top-k elements by BM25 score against `query`, in the
    same {element_id, kind, page, source, summary} shape retrieval.py's
    vector retrieve() returns -- so both can be merged by
    reciprocal_rank_fusion() without either side needing to know which
    kind of search the other used.
    """
    bm25, element_ids = build_bm25_index(docstore, source_filter)
    if not element_ids:
        return []

    scores = bm25.get_scores(_tokenize(query))
    ranked = sorted(zip(element_ids, scores), key=lambda pair: pair[1], reverse=True)

    candidates = []
    for element_id, score in ranked[:k]:
        if score <= 0:
            continue   # BM25 scores are 0 for queries sharing no terms at all -- not a real match, don't pad the result with noise
        rec = docstore[element_id]
        candidates.append({
            "element_id": element_id, "kind": rec["kind"], "page": rec["page"],
            "source": rec["source"], "summary": rec["summary"],
        })
    return candidates
