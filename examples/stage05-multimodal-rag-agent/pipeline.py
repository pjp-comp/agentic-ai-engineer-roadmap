"""
PIPELINE -- wires retrieval.py and generation.py together into the
corrective-RAG loop this whole example demonstrates. This is the ONE
file that knows about both halves; retrieval.py and generation.py never
import each other, so the two remain independently testable (call
retrieve()/grade()/generate() directly if you want to inspect a single
stage without running the whole loop).

    retrieve -> grade -> (relevant found? -> generate)
                       -> (nothing relevant, haven't retried? -> rewrite_query -> retrieve again)
                       -> (nothing relevant, already retried? -> refuse)

"retrieve" above means EITHER retrieval.retrieve() (vector-only, the
default) or retrieval.hybrid_retrieve() (vector + BM25 keyword search,
fused by rank -- see retrieval.py's module docstring), selected by
run_rag_turn's use_hybrid flag. Everything downstream of retrieval --
grading, generation, the retry loop -- is identical either way, because
both retrieval modes return candidates in the same shape.

No LangGraph StateGraph here -- deliberately a plain function loop, since
this example's actual point is the multi-vector retrieval technique, the
shared multi-PDF index, and the session integration, not re-demonstrating
the graph machinery stage05-rag-langgraph already covers in depth. The
retrieve -> grade -> (generate | rewrite -> retrieve | refuse) SHAPE is
identical either way -- see stage05-rag-langgraph if you want this same
logic expressed as StateGraph nodes and add_conditional_edges instead of
a while loop.
"""

from langchain_chroma import Chroma

from generation import generate, refuse
from retrieval import grade, hybrid_retrieve, retrieve, rewrite_query, route_after_grading


def run_rag_turn(question: str, vector_store: Chroma, docstore: dict, source_filter: str | None = None,
                  use_hybrid: bool = False) -> str:
    """source_filter=None (the default) searches across every PDF in the
    shared index at once; pass a filename to narrow retrieval to just
    that document, using the "source" tag on every indexed element.

    use_hybrid=True switches retrieval from vector-only to hybrid
    (vector + BM25 keyword search, fused by rank -- see retrieval.py).
    Off by default so the base corrective-RAG behavior this example
    started with is unchanged unless you opt in.
    """
    def do_retrieve(q: str) -> list[dict]:
        if use_hybrid:
            return hybrid_retrieve(q, vector_store, docstore, source_filter)
        return retrieve(q, vector_store, source_filter)

    query = question
    original_query = question
    retries = 0

    candidates = do_retrieve(query)
    graded_relevant = grade(candidates, original_query, docstore)

    while True:
        route = route_after_grading(graded_relevant, retries)
        if route == "generate":
            return generate(graded_relevant, original_query)
        if route == "rewrite_query":
            query = rewrite_query(original_query)
            retries += 1
            candidates = do_retrieve(query)
            graded_relevant = grade(candidates, original_query, docstore)
            continue
        print("  [refuse] no relevant content after retry -- refusing rather than guessing")
        return refuse()
