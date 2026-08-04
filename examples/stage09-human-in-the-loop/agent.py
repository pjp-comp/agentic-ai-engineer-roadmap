"""
Human-in-the-loop approval gate (Stage 9 --
docs/stage-09-human-in-the-loop.md) built on LangGraph's real
interrupt_before mechanism, exactly matching that stage's brief:

    graph = builder.compile(
        checkpointer=PostgresSaver(conn),
        interrupt_before=["execute_high_risk_action"],
    )
    graph.update_state(thread_id, {"approved": True})
    graph.invoke(None, config={"configurable": {"thread_id": thread_id}})

(SqliteSaver here instead of PostgresSaver -- same idea as stage06 and
stage05: genuine cross-process persistence without needing a running
Postgres instance for a learning example. Swap the checkpointer class and
nothing else about this file's shape changes.)

Ollama only -- same LOCAL_MODEL as every other stage03+ example.

The graph:

    START -> draft_action -> [INTERRUPT] -> execute_high_risk_action -> END

  draft_action              -- an AGENT node: the model decides WHAT
                                high-risk action to take (drafts a refund
                                amount + reason from the user's request)
                                but does not run it yet.
  execute_high_risk_action  -- runs the actual side effect (here: a fake
                                "send_refund" tool). LangGraph is told to
                                interrupt_before THIS node, so the graph
                                stops after drafting and before acting.

Why the interrupt happens where it does: Stage 9's whole point is that a
human reviews the PROPOSED action before any side effect happens, not
after. Putting the interrupt before execute_high_risk_action instead of
after it is what makes this an approval gate rather than an audit log.

Usage:
    ollama pull llama3.2:3b   # one-time, ~2GB, shared with other examples
    uv run agent.py "refund the customer $45 for a damaged item"
        -> drafts the action, then PAUSES and prints a thread_id + the
           proposed action, without executing anything
    uv run agent.py --approve <thread_id>
        -> resumes that exact run, with approval, in a NEW process --
           execute_high_risk_action actually runs now
    uv run agent.py --reject <thread_id> "reason"
        -> resumes with rejection -- the side effect never runs, the
           rejection reason is recorded in state
    uv run agent.py --list
        -> show every thread_id and its current status (pending/approved/rejected)
"""

import sqlite3
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
DB_PATH = Path(__file__).parent / ".approvals.db"


class ApprovalState(TypedDict):
    request: str
    proposed_action: str
    proposed_amount: float
    approved: bool | None   # None = awaiting a human decision
    rejection_reason: str
    executed: bool
    result: str


def make_draft_node(llm: ChatOllama):
    def draft_action(state: ApprovalState) -> dict:
        print(f"  [draft_action] reasoning about: {state['request']!r}", file=sys.stderr)
        response = llm.invoke(
            "A customer support agent needs to propose (not execute) a refund. "
            f"Request: {state['request']!r}. "
            "Reply with exactly two lines:\n"
            "AMOUNT: <a dollar figure, digits only, e.g. 45.00>\n"
            "REASON: <one short sentence>"
        )
        text = response.content
        amount = 0.0
        reason = text
        for line in text.splitlines():
            if line.upper().startswith("AMOUNT:"):
                digits = "".join(c for c in line.split(":", 1)[1] if c.isdigit() or c == ".")
                try:
                    amount = float(digits) if digits else 0.0
                except ValueError:
                    amount = 0.0
            elif line.upper().startswith("REASON:"):
                reason = line.split(":", 1)[1].strip()

        print(f"  [draft_action] proposing ${amount:.2f} -- {reason!r}", file=sys.stderr)
        return {
            "proposed_action": reason,
            "proposed_amount": amount,
            "approved": None,   # explicitly unresolved -- the interrupt happens next
        }

    return draft_action


def execute_high_risk_action(state: ApprovalState) -> dict:
    """The actual side effect. LangGraph is configured to interrupt
    BEFORE this node runs, so by the time this code executes, a human has
    already set state["approved"] via update_state().
    """
    if not state["approved"]:
        print("  [execute_high_risk_action] rejected -- not executing", file=sys.stderr)
        return {"executed": False, "result": f"Refund rejected: {state['rejection_reason']}"}

    print(f"  [execute_high_risk_action] sending refund of ${state['proposed_amount']:.2f}", file=sys.stderr)
    # A real system would call a payments API here. This is a fake side
    # effect on purpose -- the point of this example is the gate, not the
    # refund logic itself.
    return {
        "executed": True,
        "result": f"Refund of ${state['proposed_amount']:.2f} sent. Reason: {state['proposed_action']}",
    }


def build_graph(llm: ChatOllama, saver: SqliteSaver):
    graph = StateGraph(ApprovalState)
    graph.add_node("draft_action", make_draft_node(llm))
    graph.add_node("execute_high_risk_action", execute_high_risk_action)
    graph.add_edge(START, "draft_action")
    graph.add_edge("draft_action", "execute_high_risk_action")
    graph.add_edge("execute_high_risk_action", END)
    return graph.compile(checkpointer=saver, interrupt_before=["execute_high_risk_action"])


def start_request(request: str) -> None:
    llm = ChatOllama(model=LOCAL_MODEL, temperature=0)
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    print(f"  [model] local via Ollama: {LOCAL_MODEL}", file=sys.stderr)

    with SqliteSaver.from_conn_string(str(DB_PATH)) as saver:
        app = build_graph(llm, saver)
        app.invoke(
            {
                "request": request,
                "proposed_action": "",
                "proposed_amount": 0.0,
                "approved": None,
                "rejection_reason": "",
                "executed": False,
                "result": "",
            },
            config=config,
        )
        # invoke() returns here because interrupt_before paused the graph
        # BEFORE execute_high_risk_action -- nothing has run yet.
        state = app.get_state(config)

    print(f"\nProposed action: refund ${state.values['proposed_amount']:.2f}")
    print(f"Reason: {state.values['proposed_action']}")
    print(f"\n(paused for approval -- thread_id: {thread_id})")
    print(f"Approve:  uv run agent.py --approve {thread_id}")
    print(f"Reject:   uv run agent.py --reject {thread_id} \"not warranted\"")


def resume(thread_id: str, approved: bool, reason: str = "") -> None:
    """Runs in a NEW process from start_request -- this is the real test
    of the pattern: the decision that unpauses the graph doesn't have to
    come from the same run, or even the same machine, as long as it has
    the checkpointer's storage and the thread_id.
    """
    llm = ChatOllama(model=LOCAL_MODEL, temperature=0)
    config = {"configurable": {"thread_id": thread_id}}

    with SqliteSaver.from_conn_string(str(DB_PATH)) as saver:
        app = build_graph(llm, saver)
        existing = app.get_state(config)
        if not existing.values:
            print(f"No pending request found for thread_id={thread_id}.")
            return
        if existing.values.get("executed") or existing.values.get("approved") is False:
            print(f"thread_id={thread_id} was already resolved -- nothing to do.")
            return

        # This is the brief's graph.update_state(...) call: inject the
        # human decision into the paused graph's state.
        app.update_state(config, {"approved": approved, "rejection_reason": reason})
        # This is the brief's graph.invoke(None, config=...) call: resume
        # from exactly where interrupt_before paused, no re-run of draft_action.
        result = app.invoke(None, config=config)

    print(f"\n{result['result']}")


def list_requests() -> None:
    if not DB_PATH.exists():
        print(f"No {DB_PATH.name} found yet -- start a request first.")
        return
    llm = ChatOllama(model=LOCAL_MODEL, temperature=0)
    conn = sqlite3.connect(str(DB_PATH))
    thread_ids = [row[0] for row in conn.execute("SELECT DISTINCT thread_id FROM checkpoints").fetchall()]
    conn.close()

    if not thread_ids:
        print("No requests recorded yet.")
        return

    with SqliteSaver.from_conn_string(str(DB_PATH)) as saver:
        app = build_graph(llm, saver)
        print(f"{len(thread_ids)} request(s):")
        for tid in thread_ids:
            state = app.get_state({"configurable": {"thread_id": tid}}).values
            if state.get("executed"):
                status = "approved+executed"
            elif state.get("approved") is False:
                status = "rejected"
            else:
                status = "pending approval"
            print(f"  {tid}  [{status}]  ${state.get('proposed_amount', 0):.2f} -- {state.get('proposed_action', '')!r}")


if __name__ == "__main__":
    argv = sys.argv[1:]
    if argv and argv[0] == "--approve":
        resume(argv[1], approved=True)
    elif argv and argv[0] == "--reject":
        resume(argv[1], approved=False, reason=" ".join(argv[2:]) or "rejected")
    elif argv and argv[0] == "--list":
        list_requests()
    else:
        request = " ".join(argv) or "refund the customer $45 for a damaged item"
        start_request(request)
