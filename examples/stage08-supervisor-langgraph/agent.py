"""
Multi-agent supervisor orchestration (Stage 8 --
docs/stage-08-multi-agent.md) as the exact graph shape that stage's brief
describes -- a supervisor node that ROUTES (deterministic code, no LLM
call) between two worker agents, workers that write results back to
shared state and never call each other directly, and a loop back to the
supervisor after every worker turn so it can decide what happens next.

Ollama only -- same LOCAL_MODEL as every other stage03/04/05/06 example.

The graph:

    START -> supervisor -> (route_next) -> researcher -> supervisor
                                |-> writer -> supervisor
                                |-> done -> END

  supervisor  - a ROUTER node (Stage 8's node-type table): plain Python,
               no LLM call, deciding "who goes next" purely from what's
               already in state. Doing this in code instead of asking an
               LLM "who should go next?" is the cost saving Stage 8 calls
               out explicitly: a router doesn't need to reason.
  researcher  - an AGENT node: one LLM call, writes research_notes and
               research_done into shared state.
  critic      - a GUARDRAIL node, sitting between researcher and writer at
               the handoff (Stage 8's "Guardrails at the handoff, not just
               at the end" section) -- rejects an empty or too-short
               research result and routes back to the researcher instead
               of letting the writer draft from garbage.
  writer      - an AGENT node: drafts from research_notes, never calls
               researcher directly -- its only inputs are graph state and
               its own code, same structural point stage06's example
               makes about inter-agent communication.

Why a supervisor instead of a fixed researcher-then-writer chain: a fixed
chain can't express "researcher failed, try again" or "we're done, stop"
without special-casing it outside the graph. Routing through one
supervisor node makes every "what happens next" decision a single,
inspectable place (route_next) instead of scattered edge conditions.

Usage:
    ollama pull llama3.2:3b   # one-time, ~2GB, shared with other examples
    uv run agent.py "the tradeoffs of microservices vs a monolith"
"""

import sys
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

LOCAL_MODEL = "llama3.2:3b"
MAX_RESEARCH_ATTEMPTS = 2   # graceful degradation, same idea as stage05's MAX_RETRIES


class SupervisorState(TypedDict):
    topic: str
    research_notes: str
    research_done: bool
    research_attempts: int
    critic_verdict: str      # "" | "approved" | "rejected"
    draft: str
    next_step: str           # what route_next last decided -- inspectable, not hidden


def make_researcher_node(llm: ChatOllama):
    def researcher_node(state: SupervisorState) -> dict:
        print(f"  [researcher] researching {state['topic']!r} (attempt {state['research_attempts'] + 1})", file=sys.stderr)
        response = llm.invoke(
            f"Write 2-3 short bullet-point research notes about: {state['topic']}. "
            "Keep it factual and brief -- this is a research summary, not a full report."
        )
        return {
            "research_notes": response.content,
            "research_done": True,
            "research_attempts": state["research_attempts"] + 1,
            "critic_verdict": "",   # reset -- critic hasn't looked at this attempt yet
        }

    return researcher_node


def critic_node(state: SupervisorState) -> dict:
    """The guardrail-at-the-handoff node: a cheap, non-LLM check between
    researcher and writer. Rejects an empty or suspiciously short result
    instead of letting the writer draft from garbage (Stage 8: "Bad input
    propagates unchecked" if nothing sits at this boundary).
    """
    notes = state["research_notes"].strip()
    if len(notes) < 20:
        print(f"  [critic] rejected -- research_notes too short ({len(notes)} chars)", file=sys.stderr)
        return {"critic_verdict": "rejected"}
    print("  [critic] approved -- handoff to writer", file=sys.stderr)
    return {"critic_verdict": "approved"}


def make_writer_node(llm: ChatOllama):
    def writer_node(state: SupervisorState) -> dict:
        # Structurally the same point as stage06: writer_node never calls
        # researcher_node. It only reads state the researcher (and critic)
        # already wrote.
        print("  [writer] drafting from approved research notes", file=sys.stderr)
        response = llm.invoke(
            f"Using these research notes, write a 2-sentence summary:\n\n{state['research_notes']}"
        )
        return {"draft": response.content}

    return writer_node


def route_next(state: SupervisorState) -> str:
    """The supervisor's entire decision, as a plain function returning a
    string -- Stage 8's ROUTER node type. No LLM call: everything it needs
    is already in state.
    """
    if state["draft"]:
        return "done"
    if not state["research_done"]:
        return "researcher"
    if state["critic_verdict"] == "rejected":
        if state["research_attempts"] >= MAX_RESEARCH_ATTEMPTS:
            # Graceful degradation: don't loop forever on research that
            # keeps failing the critic -- hand the writer whatever exists
            # rather than spin. Real systems might escalate to a human
            # here instead (Stage 9).
            print("  [supervisor] max research attempts hit -- forwarding anyway", file=sys.stderr)
            return "writer"
        return "researcher"
    if state["critic_verdict"] == "approved":
        return "writer"
    return "researcher"


def supervisor_node(state: SupervisorState) -> dict:
    """The supervisor itself does no work beyond deciding + recording that
    decision -- route_next is called separately by add_conditional_edges,
    this node just makes the choice visible in state for observability
    (Stage 11 will want this: which node did the supervisor pick, and
    why, per run).
    """
    decision = route_next(state)
    print(f"  [supervisor] routing -> {decision}", file=sys.stderr)
    return {"next_step": decision}


def build_graph(llm: ChatOllama):
    graph = StateGraph(SupervisorState)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("researcher", make_researcher_node(llm))
    graph.add_node("critic", critic_node)
    graph.add_node("writer", make_writer_node(llm))

    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        route_next,
        {"researcher": "researcher", "writer": "writer", "done": END},
    )
    # researcher's result always passes through the critic before the
    # supervisor sees it again -- the handoff guardrail sits ON the edge,
    # not buried inside either agent.
    graph.add_edge("researcher", "critic")
    graph.add_edge("critic", "supervisor")
    graph.add_edge("writer", "supervisor")
    return graph.compile()


def run(topic: str) -> None:
    llm = ChatOllama(model=LOCAL_MODEL, temperature=0)
    print(f"  [model] local via Ollama: {LOCAL_MODEL}", file=sys.stderr)

    app = build_graph(llm)
    result = app.invoke(
        {
            "topic": topic,
            "research_notes": "",
            "research_done": False,
            "research_attempts": 0,
            "critic_verdict": "",
            "draft": "",
            "next_step": "",
        },
        config={"recursion_limit": 25},
    )

    print(f"\n{result['draft']}")


if __name__ == "__main__":
    topic = " ".join(sys.argv[1:]) or "the tradeoffs of microservices vs a monolith"
    run(topic)
