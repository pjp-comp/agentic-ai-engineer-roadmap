"""
Agentic, multimodal RAG over real PDFs (Stage 5's RAG patterns
(docs/stage-05-rag-retrieval.md) + Stage 6's Session/Event pattern
(docs/stage-06-sessions-state.md), combined). This file is the CLI + the
session-orchestration layer ONLY -- every actual pipeline stage lives in
its own file, described below. Read those files' docstrings for the real
mechanics; this one is deliberately thin.

Module map, in the order data flows through them:

  chunking.py      -- PDF -> typed Elements (text/table/image), each
                       tagged with page + source filename.
  vision.py         -- Elements -> text summaries. Tables/images need a
                       model call each (VISION_MODEL for images -- this
                       is the ONE stage that must be vision-capable).
  embeddings.py     -- which embedding model turns summaries into
                       vectors (+ an optional CLIP path, off by default
                       -- see that file for why).
  vector_store.py   -- owns the Chroma collection (.chroma_multimodal/,
                       ONE SHARED store across every indexed PDF) and
                       docstore.json (raw content, looked up by
                       element_id).
  keyword_search.py  -- BM25 keyword search over the same summaries, for
                       hybrid retrieval (--hybrid flag) -- catches exact
                       terms (region names, figures) vector search alone
                       can miss.
  retrieval.py      -- retrieve (vector search over summaries) + grade
                       (is each candidate ACTUALLY useful?) +
                       rewrite_query (retry once on a weak first pass) +
                       hybrid_retrieve (vector + keyword, fused by rank).
  generation.py     -- builds the final answer from graded-relevant RAW
                       content, including a real fix for table-sum
                       arithmetic (see that file's compute_table_totals).
  pipeline.py        -- wires retrieval.py + generation.py into the
                       corrective-RAG loop: retrieve -> grade ->
                       (generate | rewrite_query -> retrieve again | refuse).
  session.py         -- Session/Event/PersistentSessionService, so every
                       question-answer turn is recorded and a research
                       conversation survives a restart.
  build_index.py     -- the one-time-per-PDF indexing command (run this
                       BEFORE agent.py). PDFs live in assets/.
  agent.py (here)    -- CLI parsing, session load/resume, calls
                       pipeline.run_rag_turn() per question.

PDFs live in assets/ -- this example does not ship a PDF generator or any
sample PDFs; drop your own file(s) in there and index them with
build_index.py --pdf <filename>.

Usage:
    ollama pull llama3.1:8b        # one-time, ~4.9GB, shared with other examples
    ollama pull qwen2.5vl:7b        # one-time, ~6GB, vision model for images
    ollama pull nomic-embed-text    # one-time, ~274MB, shared with stage04/stage05
    # drop a PDF into assets/, then:
    uv run build_index.py --pdf your_file.pdf
    uv run agent.py "a question about the PDF"
        -> searches across EVERY indexed PDF, prints an answer + a session id
    uv run agent.py --source your_file.pdf "..."
        -> narrow retrieval to just one indexed document, when you know
           which one the answer should come from
    uv run agent.py --hybrid "a question with exact terms/figures in it"
        -> vector search fused with BM25 keyword search (Reciprocal Rank
           Fusion) instead of vector-only. Composes with --source,
           --resume, and --chat. See retrieval.py for why hybrid catches
           things vector-only search can miss.
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

from dotenv import load_dotenv

from pipeline import run_rag_turn
from session import PersistentSessionService, Session
from vector_store import PERSIST_DIR, load_docstore, open_vector_store
from vision import LOCAL_MODEL, VISION_MODEL

load_dotenv(Path(__file__).resolve().parents[2] / ".env")


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
                  vector_store, docstore: dict, source_filter: str | None = None,
                  use_hybrid: bool = False) -> str:
    """One question -> one answer -> one recorded turn. Pulled out of
    ask() so chat_loop() can call it repeatedly against the SAME session
    object, instead of ask()'s one-shot version reloading/re-fetching a
    session from disk on every single question.
    """
    session.emit("user_message", {"text": question})
    service.save(session)

    answer = run_rag_turn(question, vector_store, docstore, source_filter, use_hybrid)

    session.emit("agent_message", {"from": "rag_agent", "text": "Answered from document."})
    session.update_context(last_question=question, last_answer=answer)
    service.save(session)
    return answer


def _load_pipeline_inputs() -> tuple:
    print(f"  [model] local via Ollama: {LOCAL_MODEL} (chat/grading), {VISION_MODEL} (vision, index-time only)", file=sys.stderr)
    vector_store = open_vector_store()
    docstore = load_docstore(PERSIST_DIR)
    if not docstore:
        print("No index found. Drop a PDF in assets/ and run `uv run build_index.py --pdf <filename>` first.")
        return None, None
    return vector_store, docstore


def ask(question: str, session_id: str | None = None, source_filter: str | None = None,
        use_hybrid: bool = False) -> None:
    """One question, one process, one exit -- the scripting-friendly
    path. For a real back-and-forth without re-paying startup cost or
    retyping --resume every time, use --chat instead (see chat_loop()).
    """
    vector_store, docstore = _load_pipeline_inputs()
    if docstore is None:
        return

    service = PersistentSessionService()
    session = _load_session(session_id, service)
    if session is None:
        return

    answer = _answer_turn(question, session, service, vector_store, docstore, source_filter, use_hybrid)

    print(f"\n{answer}")
    print(f"\n(session id: {session.id})")
    print(f"Ask a follow-up in THIS session: uv run agent.py --resume {session.id} \"...\"")
    print(f"Or keep going without restarting: uv run agent.py --chat {session.id}")


def chat_loop(session_id: str | None = None, source_filter: str | None = None, use_hybrid: bool = False) -> None:
    """Continue the SAME session across multiple questions in ONE
    process -- this is what --resume alone doesn't give you: --resume
    starts a brand-new `uv run agent.py` invocation per question, which
    means re-opening the vector store, re-loading the docstore, and
    re-fetching the session from disk every single time, plus retyping
    the session id on every call. Here, all three are loaded ONCE, then
    every line typed at the prompt becomes another turn against the same
    already-loaded session, same as a real chat UI would do it.
    """
    vector_store, docstore = _load_pipeline_inputs()
    if docstore is None:
        return

    service = PersistentSessionService()
    session = _load_session(session_id, service)
    if session is None:
        return

    scope = source_filter or "all indexed documents"
    mode = "hybrid (vector + keyword)" if use_hybrid else "vector-only"
    print(f"\nChatting about {scope} -- session {session.id} -- retrieval: {mode}")
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

        answer = _answer_turn(question, session, service, vector_store, docstore, source_filter, use_hybrid)
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

    # --source and --hybrid compose with every mode below (--resume,
    # --chat, or the default single question) -- pulled out first so
    # each branch doesn't need to re-implement the same parsing.
    # source_filter=None means "search every indexed PDF" (the default,
    # shared-index behavior). use_hybrid=False means vector-only
    # retrieval (the default) -- --hybrid opts into vector + BM25
    # keyword search fused by rank (see retrieval.py).
    source_filter = None
    if "--source" in argv:
        idx = argv.index("--source")
        source_filter = argv[idx + 1]
        argv = argv[:idx] + argv[idx + 2:]

    use_hybrid = "--hybrid" in argv
    if use_hybrid:
        argv = [a for a in argv if a != "--hybrid"]

    if argv and argv[0] == "--history":
        print_history(argv[1] if len(argv) > 1 else "")
    elif argv and argv[0] == "--list":
        list_sessions()
    elif argv and argv[0] == "--resume":
        session_id = argv[1] if len(argv) > 1 else ""
        question = " ".join(argv[2:]) or "Summarize the document."
        ask(question, session_id=session_id, source_filter=source_filter, use_hybrid=use_hybrid)
    elif argv and argv[0] == "--chat":
        session_id = argv[1] if len(argv) > 1 else None
        chat_loop(session_id=session_id, source_filter=source_filter, use_hybrid=use_hybrid)
    elif argv:
        question = " ".join(argv)
        ask(question, source_filter=source_filter, use_hybrid=use_hybrid)
    else:
        print("Usage: uv run agent.py \"a question\"")
        print("       uv run agent.py --chat | --resume <id> | --history <id> | --list")
        print("       Add --hybrid for vector+keyword fused retrieval, --source <file> to narrow to one PDF")
