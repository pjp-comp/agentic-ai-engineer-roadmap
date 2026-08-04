"""
Session, state, and events (Stage 6 — docs/stage-06-sessions-state.md) as
real LangGraph mechanics instead of the hand-rolled Session/Event classes
in that doc. Two agents (a researcher and a writer) communicate ONLY
through shared graph state, exactly like the doc's researcher_turn/
writer_turn example — this file is that same idea built on LangGraph's
actual session primitive instead of a from-scratch dataclass.

Ollama only — same LOCAL_MODEL as every other stage03/04/05 example.

How LangGraph maps onto Stage 6's three concepts:

  Session   -> `thread_id` in the config passed to app.invoke(). LangGraph's
              checkpointer scopes ALL state to a thread_id, the same way
              Stage 6's Session.id scopes a run. Two different thread_ids
              are two completely independent sessions, even against the
              same compiled graph.
  State     -> the TypedDict state schema below (SessionState) — "what do
              we currently know right now." Every node returns a partial
              update; LangGraph merges it into the current state.
  Events    -> LangGraph's checkpointer keeps a full history of every
              state version (app.get_state_history()) — a trail of STATE
              SNAPSHOTS, not a hand-written append-only event log with
              named event types. This file adds an explicit
              `events: list[dict]` field to the state schema itself, so
              "what happened, in order, with a type" stays a visible,
              inspectable list the way Stage 6's Event class makes it —
              not something you'd have to reconstruct by diffing snapshots.

Why SqliteSaver, not InMemorySaver: InMemorySaver only lives as long as the
Python process does — a session "persisting" only within one `uv run`
invocation isn't really demonstrating persistence, it's just a variable
staying in scope. SqliteSaver writes checkpoints to a real .sqlite file, so
--resume/--history below work as genuinely separate process runs, the same
way Stage 4's PersistentState survives a restart.

Why two agents, one graph, no direct connection: `writer_node` never calls
`researcher_node`. It only reads `state["research_done"]`. This is Stage
6's actual point about inter-agent communication — agents share a session,
they don't call each other — made unavoidable by how LangGraph nodes work:
a node's only inputs are the graph state and its own code.

Usage:
    ollama pull llama3.2:3b   # one-time, ~2GB, shared with other examples
    uv run agent.py "Northwind Traders Q3 results"
        -> prints a thread_id at the end; every session gets a new one
    uv run agent.py --history <thread_id>
        -> re-open the SAME session in a NEW process and print its full
           state + event log, proving it was written to disk, not held
           in memory
    uv run agent.py --list
        -> list all thread_ids ever run against .sessions.db in this directory
"""

import sys
import uuid
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

LOCAL_MODEL = "llama3.2:3b"
DB_PATH = Path(__file__).parent / ".sessions.db"


class SessionState(TypedDict):
    topic: str
    research_notes: str
    research_done: bool
    draft: str
    step: str
    events: list[dict]   # append-only log — Stage 6's Event, as a plain list


def _emit(state: SessionState, event_type: str, data: dict) -> list[dict]:
    """Stage 6's Session.emit(), expressed as a state-update helper instead
    of a mutating method — LangGraph nodes return updates, they don't
    mutate state in place, so "append-only" here means "return the old
    list plus one more item," not "call .append() on shared state."
    """
    return state["events"] + [{"type": event_type, "data": data}]


def make_researcher_node(llm: ChatOllama):
    def researcher_node(state: SessionState) -> dict:
        print(f"  [researcher] researching {state['topic']!r}", file=sys.stderr)
        response = llm.invoke(
            f"Write 2-3 short bullet-point research notes about: {state['topic']}. "
            "Keep it brief — this is a research summary, not a full report."
        )
        events = _emit(state, "agent_message", {"from": "researcher", "text": "Research complete."})
        return {
            "research_notes": response.content,
            "research_done": True,
            "step": "researched",
            "events": events,
        }

    return researcher_node


def make_writer_node(llm: ChatOllama):
    def writer_node(state: SessionState) -> dict:
        # This is the entire point of the example: the writer does NOT call
        # the researcher. It reads state["research_done"] — a value the
        # researcher wrote to the SHARED session — and only proceeds
        # because it can see that flag, the same way Stage 6's writer_turn
        # checks session.state.get("research_done") before drafting.
        if not state["research_done"]:
            print("  [writer] research_done is False — waiting, not drafting", file=sys.stderr)
            return {"step": "waiting_on_research"}

        print("  [writer] drafting from researcher's notes", file=sys.stderr)
        response = llm.invoke(
            f"Using these research notes, write a 2-sentence summary:\n\n{state['research_notes']}"
        )
        events = _emit(state, "agent_message", {"from": "writer", "text": "Draft complete."})
        return {"draft": response.content, "step": "done", "events": events}

    return writer_node


def build_graph(llm: ChatOllama, saver: SqliteSaver):
    graph = StateGraph(SessionState)
    graph.add_node("researcher", make_researcher_node(llm))
    graph.add_node("writer", make_writer_node(llm))
    graph.add_edge(START, "researcher")
    graph.add_edge("researcher", "writer")
    graph.add_edge("writer", END)
    # The checkpointer is what turns thread_id into a real session: every
    # invoke() against the same thread_id resumes from where that thread's
    # state last left off, instead of starting fresh each call.
    return graph.compile(checkpointer=saver)


def run_session(topic: str) -> None:
    llm = ChatOllama(model=LOCAL_MODEL, temperature=0)
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    print(f"  [model] local via Ollama: {LOCAL_MODEL}", file=sys.stderr)
    print(f"  [session] new thread_id={thread_id}", file=sys.stderr)

    with SqliteSaver.from_conn_string(str(DB_PATH)) as saver:
        app = build_graph(llm, saver)
        result = app.invoke(
            {
                "topic": topic,
                "research_notes": "",
                "research_done": False,
                "draft": "",
                "step": "start",
                "events": [],
            },
            config=config,
        )

    print(f"\n{result['draft']}")
    print(f"\n(session thread_id: {thread_id})")
    print(f"Inspect it later, in a NEW process: uv run agent.py --history {thread_id}")


def print_history(thread_id: str) -> None:
    """Re-open a session by thread_id in a fresh process — proves state
    genuinely persisted to .sessions.db, not just stayed in a variable.
    """
    if not DB_PATH.exists():
        print(f"No {DB_PATH.name} found yet — run a session first.")
        return
    llm = ChatOllama(model=LOCAL_MODEL, temperature=0)
    with SqliteSaver.from_conn_string(str(DB_PATH)) as saver:
        app = build_graph(llm, saver)
        config = {"configurable": {"thread_id": thread_id}}
        state = app.get_state(config)

    if not state.values:
        print(f"No session found for thread_id={thread_id}.")
        return

    print(f"Final state for thread_id={thread_id}:")
    for key, value in state.values.items():
        if key == "events":
            continue
        print(f"  {key}: {value!r}")
    print("\nEvent log (append-only, in order):")
    for i, event in enumerate(state.values.get("events", [])):
        print(f"  [{i}] {event['type']}: {event['data']}")


def list_sessions() -> None:
    """List every distinct thread_id ever checkpointed in .sessions.db —
    a crude but real "which sessions exist" query, using the checkpointer's
    own storage instead of a separate session index (Stage 6's
    InMemorySessionService keeps a dict of sessions for exactly this
    purpose; SqliteSaver's checkpoints table already has the equivalent
    information, so this reads that directly instead of duplicating it).
    """
    if not DB_PATH.exists():
        print(f"No {DB_PATH.name} found yet — run a session first.")
        return
    import sqlite3

    conn = sqlite3.connect(str(DB_PATH))
    rows = conn.execute("SELECT DISTINCT thread_id FROM checkpoints").fetchall()
    conn.close()
    if not rows:
        print("No sessions recorded yet.")
        return
    print(f"{len(rows)} session(s) recorded in {DB_PATH.name}:")
    for (thread_id,) in rows:
        print(f"  {thread_id}")


if __name__ == "__main__":
    argv = sys.argv[1:]
    if argv and argv[0] == "--history":
        print_history(argv[1] if len(argv) > 1 else "")
    elif argv and argv[0] == "--list":
        list_sessions()
    else:
        topic = " ".join(argv) or "Northwind Traders Q3 results"
        run_session(topic)
