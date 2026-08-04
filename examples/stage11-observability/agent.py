"""
Stage 11's brief, applied literally: "Instrument the Stage 8 supervisor
graph with tracing so every node emits a span with input/output, token
counts, latency, and cost."

This file IS the Stage 8 supervisor graph (examples/stage08-supervisor-langgraph/agent.py)
with one addition: every node is wrapped in tracer.span(...), so a run
produces a full trace instead of just printing progress to stderr. Same
graph shape, same nodes, same routing logic -- read stage08's README
first if the graph itself (router/critic/agent nodes) isn't already
familiar; this file's only new content is the tracing layer.

Ollama only -- same LOCAL_MODEL as stage08.

Usage:
    ollama pull llama3.2:3b   # one-time, ~2GB, shared with other examples
    uv run agent.py "the tradeoffs of microservices vs a monolith"
        -> runs the graph, writes one span per node to .traces.jsonl,
           prints the trace_id at the end
    uv run dashboard.py
        -> reads .traces.jsonl, prints p95 latency and $/run per node
           and per trace -- Stage 11's "one dashboard view" requirement
    uv run dashboard.py --trace <trace_id>
        -> Stage 11's "Done when": given a bad output, find the exact
           span that caused it in under two minutes -- this is that
           lookup, given a trace_id from a run's output
"""

import sys
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph

from tracing import Tracer

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

LOCAL_MODEL = "llama3.2:3b"
MAX_RESEARCH_ATTEMPTS = 2


class SupervisorState(TypedDict):
    topic: str
    research_notes: str
    research_done: bool
    research_attempts: int
    critic_verdict: str
    draft: str
    next_step: str


def _token_counts(response) -> tuple[int, int]:
    """ChatOllama surfaces Ollama's real token counts in response_metadata
    -- pulling them out here is what makes the span's token/cost fields
    real numbers instead of placeholders.
    """
    meta = getattr(response, "response_metadata", {}) or {}
    return meta.get("prompt_eval_count", 0), meta.get("eval_count", 0)


def make_researcher_node(llm: ChatOllama, tracer: Tracer):
    def researcher_node(state: SupervisorState) -> dict:
        with tracer.span("researcher", input_data=state["topic"]) as span:
            print(f"  [researcher] researching {state['topic']!r} (attempt {state['research_attempts'] + 1})", file=sys.stderr)
            response = llm.invoke(
                f"Write 2-3 short bullet-point research notes about: {state['topic']}. "
                "Keep it factual and brief -- this is a research summary, not a full report."
            )
            span.prompt_tokens, span.completion_tokens = _token_counts(response)
            span.output = response.content
            return {
                "research_notes": response.content,
                "research_done": True,
                "research_attempts": state["research_attempts"] + 1,
                "critic_verdict": "",
            }

    return researcher_node


def make_critic_node(tracer: Tracer):
    def critic_node(state: SupervisorState) -> dict:
        with tracer.span("critic", input_data=state["research_notes"]) as span:
            notes = state["research_notes"].strip()
            if len(notes) < 20:
                print(f"  [critic] rejected -- research_notes too short ({len(notes)} chars)", file=sys.stderr)
                span.output = "rejected"
                return {"critic_verdict": "rejected"}
            print("  [critic] approved -- handoff to writer", file=sys.stderr)
            span.output = "approved"
            return {"critic_verdict": "approved"}

    return critic_node


def make_writer_node(llm: ChatOllama, tracer: Tracer):
    def writer_node(state: SupervisorState) -> dict:
        with tracer.span("writer", input_data=state["research_notes"]) as span:
            print("  [writer] drafting from approved research notes", file=sys.stderr)
            response = llm.invoke(
                f"Using these research notes, write a 2-sentence summary:\n\n{state['research_notes']}"
            )
            span.prompt_tokens, span.completion_tokens = _token_counts(response)
            span.output = response.content
            return {"draft": response.content}

    return writer_node


def route_next(state: SupervisorState) -> str:
    if state["draft"]:
        return "done"
    if not state["research_done"]:
        return "researcher"
    if state["critic_verdict"] == "rejected":
        if state["research_attempts"] >= MAX_RESEARCH_ATTEMPTS:
            print("  [supervisor] max research attempts hit -- forwarding anyway", file=sys.stderr)
            return "writer"
        return "researcher"
    if state["critic_verdict"] == "approved":
        return "writer"
    return "researcher"


def make_supervisor_node(tracer: Tracer):
    def supervisor_node(state: SupervisorState) -> dict:
        with tracer.span("supervisor", input_data=state.get("critic_verdict", "")) as span:
            decision = route_next(state)
            print(f"  [supervisor] routing -> {decision}", file=sys.stderr)
            span.output = decision
            return {"next_step": decision}

    return supervisor_node


def build_graph(llm: ChatOllama, tracer: Tracer):
    graph = StateGraph(SupervisorState)
    graph.add_node("supervisor", make_supervisor_node(tracer))
    graph.add_node("researcher", make_researcher_node(llm, tracer))
    graph.add_node("critic", make_critic_node(tracer))
    graph.add_node("writer", make_writer_node(llm, tracer))

    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        route_next,
        {"researcher": "researcher", "writer": "writer", "done": END},
    )
    graph.add_edge("researcher", "critic")
    graph.add_edge("critic", "supervisor")
    graph.add_edge("writer", "supervisor")
    return graph.compile()


def run(topic: str) -> None:
    llm = ChatOllama(model=LOCAL_MODEL, temperature=0)
    tracer = Tracer()
    trace_id = tracer.start_trace()
    print(f"  [model] local via Ollama: {LOCAL_MODEL}", file=sys.stderr)
    print(f"  [trace] trace_id={trace_id}", file=sys.stderr)

    app = build_graph(llm, tracer)
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
    print(f"\n(trace_id: {trace_id})")
    print(f"Inspect it: uv run dashboard.py --trace {trace_id}")


if __name__ == "__main__":
    topic = " ".join(sys.argv[1:]) or "the tradeoffs of microservices vs a monolith"
    run(topic)
