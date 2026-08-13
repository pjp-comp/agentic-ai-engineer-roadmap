"""
Agentic, multimodal RAG over a real PDF (Stage 5's RAG patterns
(docs/stage-05-rag-retrieval.md) + Stage 6's Session/Event pattern
(docs/stage-06-sessions-state.md) + Stage 7/8's agentic-graph shape,
combined) -- the "everything above" example: a real PDF containing text,
a real table, and a real chart image, indexed with MULTI-VECTOR retrieval
(see index.py's docstring for why that's the current best-practice
technique for tables/images specifically), queried through a corrective-
RAG LangGraph graph (retrieve -> grade -> generate/refuse, same shape as
stage05-rag-langgraph), with every question-answer turn persisted as a
Session (same PersistentSessionService as stage06-sessions-plain) so a
multi-turn research conversation over one document survives a restart
and can be resumed.

Ollama only:
  LOCAL_MODEL     llama3.1:8b     -- generation, grading, table summarization
  VISION_MODEL    llava:7b        -- image/chart description (index.py only)
  EMBEDDING_MODEL nomic-embed-text -- turns summaries into vectors

Pipeline, end to end:

  ONE-TIME, PER PDF (build_index.py):
    a_document.pdf -> ingest.parse_pdf() -> [Element, Element, ...]
                    -> index.build_multi_vector_index()
                         -> summarize each element (tables/images via a
                            model call, text elements directly)
                         -> embed each SUMMARY -> Chroma (.chroma_multimodal/,
                            a SINGLE SHARED index -- every PDF you index
                            gets ADDED to the same store, distinguished by
                            a "source" field, not split into separate
                            per-document indexes)
                         -> merge raw content -> docstore.json

  EVERY QUERY (agent.py, this file):
    question -> retrieve (vector search over SUMMARIES, across every
                indexed PDF at once by default -- optionally narrowed to
                one PDF with --source)
             -> grade (is each retrieved element actually relevant?)
             -> generate (answer using RAW content, not summaries --
                see make_generate_node) OR refuse
             -> the whole turn recorded into a persisted Session

Why raw content at generate-time, not the summary that was searched: the
summary is a LOSSY compression optimized for matching a QUESTION's
phrasing, not for actually answering it -- e.g. a table summary says
"shows adoption rates by region" but the actual NUMBERS live only in the
raw table text. Retrieval and generation deliberately read different
representations of the same element for this reason (see index.py).

Usage:
    ollama pull llama3.1:8b        # one-time, ~4.9GB, shared with other examples
    ollama pull llava:7b            # one-time, ~4.7GB, vision model for images
    ollama pull nomic-embed-text    # one-time, ~274MB, shared with stage04/stage05
    uv run build_index.py                          # index sample.pdf (see that file)
    uv run build_index.py --pdf sample_complex.pdf  # ADD a second PDF to the same index
    uv run agent.py "What was East Asia's year-over-year change in renewable adoption?"
        -> searches across EVERY indexed PDF, prints an answer + a session id
    uv run agent.py --source sample_complex.pdf "..."
        -> narrow retrieval to just one indexed document, when you know
           which one the answer should come from
    uv run agent.py --resume <session_id> "a follow-up question"
        -> continue the SAME research session, same document, but as a
           NEW process each time -- fine for scripting, tedious for a
           real back-and-forth (re-pays Python/model-connection startup
           every call, and you have to retype --resume <id> each time)
    uv run agent.py --chat
        -> interactive loop: start (or --chat <session_id> to resume) a
           session ONCE, then keep typing questions at a prompt in the
           SAME process. This is "continue the same session in a loop" --
           the vector store, docstore, and session are all loaded once
           and reused turn to turn, exactly like a real chat UI would.
           Type 'exit' or 'quit' to stop.
    uv run agent.py --history <session_id>
    uv run agent.py --list
"""

import sys
from pathlib import Path
from typing import TypedDict

import ollama
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from build_index import PERSIST_DIR
from index import EMBEDDING_MODEL, LOCAL_MODEL, load_docstore
from session import PersistentSessionService, Session

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

RETRIEVE_K = 4
MAX_RETRIES = 1


class RAGState(TypedDict):
    query: str
    original_query: str
    candidates: list[dict]        # retrieved {element_id, kind, page, summary} dicts
    graded_relevant: list[dict]   # candidates that passed grading, WITH raw content attached
    retries: int
    answer: str


def _build_vector_store() -> Chroma:
    if not PERSIST_DIR.exists():
        print(
            f"  [vector store] WARNING: no index found at {PERSIST_DIR.name}/ -- "
            "run `uv run build_index.py` first. Nothing to search against.",
            file=sys.stderr,
        )
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    return Chroma(
        collection_name="multimodal_rag",
        embedding_function=embeddings,
        persist_directory=str(PERSIST_DIR),
    )


def retrieve(state: RAGState, vector_store: Chroma, source_filter: str | None = None) -> dict:
    # Search over SUMMARIES (see index.py) -- this is the multi-vector
    # retrieval technique's search step: the vector matched against the
    # query is a generated description, not raw table/image content.
    # source_filter=None searches across EVERY indexed PDF at once (the
    # default -- this is the shared-index behavior); passing a filename
    # narrows the search to just that document's elements via Chroma's
    # metadata filter, using the "source" tag every element carries.
    where = {"source": source_filter} if source_filter else None
    results = vector_store.similarity_search(state["query"], k=RETRIEVE_K, filter=where)
    candidates = [
        {"element_id": r.metadata["element_id"], "kind": r.metadata["kind"],
         "page": r.metadata["page"], "source": r.metadata["source"], "summary": r.page_content}
        for r in results
    ]
    kinds = ", ".join(f"{c['kind']}({c['source']} p{c['page']})" for c in candidates)
    print(f"  [retrieve] query={state['query']!r} -> {len(candidates)} candidate(s): {kinds}", file=sys.stderr)
    return {"candidates": candidates}


def grade(state: RAGState, docstore: dict) -> dict:
    # Grade against the SUMMARY (cheap, no need to pull raw content for
    # elements that won't pass anyway) but against the ORIGINAL question,
    # same corrective-RAG discipline as stage05-rag-langgraph.
    graded = []
    for c in state["candidates"]:
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
                    f"Question: {state['original_query']}\n\n"
                    "Does this summary contain information that is ACTUALLY useful for answering "
                    "the question -- not just a vaguely related topic or section title? "
                    "Answer no if it's just a heading, a generic intro sentence, or off-topic. "
                    "Reply with exactly one word: yes or no."
                ),
            }],
            think=False,
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
    return {"graded_relevant": graded}


def route_after_grading(state: RAGState) -> str:
    if state["graded_relevant"]:
        return "generate"
    if state["retries"] < MAX_RETRIES:
        return "rewrite_query"
    return "refuse"


def rewrite_query(state: RAGState) -> dict:
    rewritten = ollama.chat(
        model=LOCAL_MODEL,
        messages=[{
            "role": "user",
            "content": (
                f"Original question: {state['original_query']}\n\n"
                "Rewrite this as a different, more specific search query that might match "
                "a passage, table, or chart in a report on renewable energy adoption. "
                "Reply with ONLY the rewritten query, nothing else."
            ),
        }],
        think=False,
    )
    new_query = rewritten.message.content.strip()
    print(f"  [rewrite] {state['original_query']!r} -> {new_query!r}", file=sys.stderr)
    return {"query": new_query, "retries": state["retries"] + 1}


def generate(state: RAGState) -> dict:
    # RAW content here, not summaries -- see this file's module docstring
    # for why that distinction is the entire point of multi-vector retrieval.
    # Each chunk is tagged with its SOURCE document, not just its page --
    # necessary once the shared index holds more than one PDF, so an
    # answer can say which document it came from, not just where in "the"
    # document (there may be several).
    context = "\n\n".join(
        f"[{c['source']}, {c['kind']}, page {c['page']}]\n{c['raw']}" for c in state["graded_relevant"]
    )
    response = ollama.chat(
        model=LOCAL_MODEL,
        messages=[{
            "role": "user",
            "content": (
                f"Context from one or more reports (tables and image descriptions included):\n{context}\n\n"
                f"Question: {state['original_query']}\n\n"
                "Answer using ONLY the context above, in 1-3 sentences. Do not repeat the raw "
                "table or list every row -- state the specific answer directly. "
                "Cite the source document and page, e.g. (sample.pdf, page 2). If the context "
                "above doesn't actually contain the answer, say plainly that the document(s) "
                "don't cover this -- do not comment on the context itself or ask for more information."
            ),
        }],
        think=False,
    )
    return {"answer": response.message.content}


def refuse(state: RAGState) -> dict:
    print("  [refuse] no relevant content after retry -- refusing rather than guessing", file=sys.stderr)
    return {
        "answer": (
            "I don't have enough information in the document to answer this question "
            "confidently, even after rewriting the query."
        )
    }


def run_rag_turn(question: str, vector_store: Chroma, docstore: dict, source_filter: str | None = None) -> str:
    """No LangGraph StateGraph here -- deliberately a plain function
    chain, since this file's actual point is the multi-vector retrieval
    technique and the session integration, not re-demonstrating the graph
    machinery stage05-rag-langgraph already covers in depth. The
    retrieve -> grade -> (generate | rewrite -> retrieve | refuse) SHAPE
    is identical either way -- see stage05-rag-langgraph if you want this
    same logic expressed as StateGraph nodes and add_conditional_edges
    instead of a while loop.

    source_filter=None (the default) searches across every PDF in the
    shared index at once; pass a filename to narrow retrieval to just
    that document, using the "source" tag on every indexed element.
    """
    state: RAGState = {
        "query": question, "original_query": question,
        "candidates": [], "graded_relevant": [], "retries": 0, "answer": "",
    }
    state.update(retrieve(state, vector_store, source_filter))
    state.update(grade(state, docstore))

    while True:
        route = route_after_grading(state)
        if route == "generate":
            state.update(generate(state))
            break
        if route == "rewrite_query":
            state.update(rewrite_query(state))
            state.update(retrieve(state, vector_store, source_filter))
            state.update(grade(state, docstore))
            continue
        state.update(refuse(state))
        break

    return state["answer"]


# --- Session integration: every turn against the PDF is recorded --------

def _load_session(session_id: str | None, service: PersistentSessionService) -> Session | None:
    """Shared by ask() and chat_loop() -- get an existing session or
    create a new one. Returns None only on a resume-by-id that fails to
    find anything (a genuine error, distinct from "starting fresh").
    """
    if session_id:
        session = service.get(session_id)
        if session is None:
            print(f"No session found for id={session_id} -- can't resume.")
            return None
        print(f"  [session] resuming id={session.id} ({len(session.events)} event(s) so far)", file=sys.stderr)
        return session
    session = service.create()
    print(f"  [session] new id={session.id}", file=sys.stderr)
    return session


def _answer_turn(question: str, session: Session, service: PersistentSessionService,
                  vector_store: Chroma, docstore: dict, source_filter: str | None = None) -> str:
    """One question -> one answer -> one recorded turn. Pulled out of
    ask() so chat_loop() can call it repeatedly against the SAME session
    object, instead of ask()'s one-shot version reloading/re-fetching a
    session from disk on every single question.
    """
    session.emit("user_message", {"text": question})
    service.save(session)

    answer = run_rag_turn(question, vector_store, docstore, source_filter)

    session.emit("agent_message", {"from": "rag_agent", "text": "Answered from document."})
    session.update_context(last_question=question, last_answer=answer)
    service.save(session)
    return answer


def ask(question: str, session_id: str | None = None, source_filter: str | None = None) -> None:
    """One question, one process, one exit -- the scripting-friendly
    path. For a real back-and-forth without re-paying startup cost or
    retyping --resume every time, use --chat instead (see chat_loop()).
    """
    print(f"  [model] local via Ollama: {LOCAL_MODEL} (chat/grading), llava:7b (vision, index-time only)", file=sys.stderr)
    vector_store = _build_vector_store()
    docstore = load_docstore(PERSIST_DIR)
    if not docstore:
        print("No index found. Run `uv run build_index.py` first.")
        return

    service = PersistentSessionService()
    session = _load_session(session_id, service)
    if session is None:
        return

    answer = _answer_turn(question, session, service, vector_store, docstore, source_filter)

    print(f"\n{answer}")
    print(f"\n(session id: {session.id})")
    print(f"Ask a follow-up in THIS session: uv run agent.py --resume {session.id} \"...\"")
    print(f"Or keep going without restarting: uv run agent.py --chat {session.id}")


def chat_loop(session_id: str | None = None, source_filter: str | None = None) -> None:
    """Continue the SAME session across multiple questions in ONE
    process -- this is what --resume alone doesn't give you: --resume
    starts a brand-new `uv run agent.py` invocation per question, which
    means re-opening the vector store, re-loading the docstore, and
    re-fetching the session from disk every single time, plus retyping
    the session id on every call. Here, all three are loaded ONCE, then
    every line typed at the prompt becomes another turn against the same
    already-loaded session, same as a real chat UI would do it.
    """
    print(f"  [model] local via Ollama: {LOCAL_MODEL} (chat/grading), llava:7b (vision, index-time only)", file=sys.stderr)
    vector_store = _build_vector_store()
    docstore = load_docstore(PERSIST_DIR)
    if not docstore:
        print("No index found. Run `uv run build_index.py` first.")
        return

    service = PersistentSessionService()
    session = _load_session(session_id, service)
    if session is None:
        return

    scope = source_filter or "all indexed documents"
    print(f"\nChatting about {scope} -- session {session.id}")
    print("Type a question, or 'exit'/'quit' to stop. Ctrl-C also works.\n")

    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break

        answer = _answer_turn(question, session, service, vector_store, docstore, source_filter)
        print(f"\n{answer}\n")

    print(f"Session {session.id} saved. Resume it later: uv run agent.py --chat {session.id}")


def print_history(session_id: str) -> None:
    service = PersistentSessionService()
    session = service.get(session_id)
    if session is None:
        print(f"No session found for id={session_id}.")
        return
    print(f"Final state for session {session_id}:")
    for key, value in session.state.items():
        print(f"  {key}: {value!r}")
    print("\nEvent log (append-only, in order):")
    for i, event in enumerate(session.events):
        print(f"  [{i}] {event.type}: {event.data}")


def list_sessions() -> None:
    service = PersistentSessionService()
    ids = service.list_ids()
    if not ids:
        print("No sessions recorded yet.")
        return
    print(f"{len(ids)} session(s):")
    for session_id in ids:
        print(f"  {session_id}")


if __name__ == "__main__":
    argv = sys.argv[1:]

    # --source composes with every mode below (--resume, --chat, or the
    # default single question) -- pulled out first so each branch
    # doesn't need to re-implement the same parsing. None means "search
    # every indexed PDF" (the default, shared-index behavior).
    source_filter = None
    if "--source" in argv:
        idx = argv.index("--source")
        source_filter = argv[idx + 1]
        argv = argv[:idx] + argv[idx + 2:]

    if argv and argv[0] == "--history":
        print_history(argv[1] if len(argv) > 1 else "")
    elif argv and argv[0] == "--list":
        list_sessions()
    elif argv and argv[0] == "--resume":
        session_id = argv[1] if len(argv) > 1 else ""
        question = " ".join(argv[2:]) or "Summarize the document."
        ask(question, session_id=session_id, source_filter=source_filter)
    elif argv and argv[0] == "--chat":
        session_id = argv[1] if len(argv) > 1 else None
        chat_loop(session_id=session_id, source_filter=source_filter)
    else:
        question = " ".join(argv) or "What was East Asia's year-over-year change in renewable adoption?"
        ask(question, source_filter=source_filter)
