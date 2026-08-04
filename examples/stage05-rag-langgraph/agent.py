"""
Corrective RAG (CRAG) as an explicit LangGraph graph — Stage 5's naive-RAG
and corrective-RAG patterns from docs/stage-05-rag-retrieval.md, built as
real nodes and edges instead of nested function calls, so the graph shape
itself (retrieve -> grade -> generate, with a conditional retry edge) is
something you can see, not just read as pseudocode.

Ollama only — no Claude path in this example (chat model AND embeddings
both run locally). Two models are used, on purpose:

  LOCAL_MODEL      llama3.2:3b — does the actual generation and the
                    "is this chunk relevant?" grading judgment.
  EMBEDDING_MODEL   nomic-embed-text — turns text into vectors for
                    similarity search. A chat model and an embedding
                    model do fundamentally different jobs; conflating
                    them is a common beginner mistake this split avoids.

The graph, concretely:

    START -> retrieve -> grade -> (any chunk relevant?)
                                        |-- yes --> generate -> END
                                        |-- no, haven't retried --> rewrite_query -> retrieve
                                        |-- no, already retried --> refuse -> END

This is "agentic RAG" from Stage 5's spectrum table applied specifically:
retrieval becomes a decision loop (search -> judge -> retry or refuse)
instead of a single fixed retrieve-then-generate pass. The corrective
part is the grading step refusing to build an answer on weak context —
same idea as Stage 3's validation-error recovery, applied to retrieval
quality instead of malformed tool arguments.

Usage:
    ollama pull llama3.2:3b        # one-time, ~2GB, shared with other examples
    ollama pull nomic-embed-text   # one-time, ~274MB, shared with stage04-longterm-memory-vectorstore
    uv run agent.py "What was Northwind Traders' revenue in fiscal 2025?"
    uv run agent.py "What is the capital of France?"   # forces the refusal path — nothing in the corpus is relevant
"""

import sys
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langgraph.graph import END, START, StateGraph

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

LOCAL_MODEL = "llama3.2:3b"
EMBEDDING_MODEL = "nomic-embed-text"
RETRIEVE_K = 3
MAX_RETRIES = 1

# --- The corpus: a small set of documents the model did NOT train on -------
# Deliberately synthetic and specific (invented company, invented numbers)
# so a correct answer can ONLY come from retrieval — there's no way for the
# model to "already know" this from training data, which makes it obvious
# when the graph is actually using retrieved context vs. guessing.

_DOCS = [
    Document(
        page_content=(
            "Northwind Traders FY2025 Annual Report Summary: Total revenue for "
            "fiscal year 2025 was $184.3 million, up 12% from $164.6 million in "
            "FY2024. Growth was driven primarily by the Pacific region, which "
            "grew 22% year-over-year."
        ),
        metadata={"source": "annual_report_fy2025.txt"},
    ),
    Document(
        page_content=(
            "Northwind Traders Q3 FY2025 Earnings Call Notes: CFO Priya Ramesh "
            "noted that operating margin improved to 14.2%, up from 11.8% a "
            "year earlier, attributing the gain to warehouse automation "
            "completed in Q2."
        ),
        metadata={"source": "q3_earnings_call.txt"},
    ),
    Document(
        page_content=(
            "Northwind Traders Executive Team: CEO is Daniel Okafor, appointed "
            "in 2022. CFO is Priya Ramesh, appointed in 2023. The company is "
            "headquartered in Portland, Oregon, and was founded in 2009."
        ),
        metadata={"source": "company_overview.txt"},
    ),
    Document(
        page_content=(
            "Northwind Traders Risk Factors (FY2025 filing excerpt): The "
            "company's Pacific region growth is concentrated in three "
            "distribution centers; disruption to any one of them could "
            "materially affect quarterly results. Currency exposure to the "
            "Japanese yen is partially hedged."
        ),
        metadata={"source": "risk_factors.txt"},
    ),
]


class RAGState(TypedDict):
    query: str
    original_query: str
    chunks: list[Document]
    graded_relevant: list[Document]
    retries: int
    answer: str


def _build_vector_store() -> Chroma:
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    store = Chroma(collection_name="crag_demo", embedding_function=embeddings)
    store.add_documents(_DOCS)
    return store


def _build_llm() -> ChatOllama:
    return ChatOllama(model=LOCAL_MODEL, temperature=0)


# --- Graph nodes -------------------------------------------------------------
# Each node is a plain function: state in, partial state update out. This is
# the same "harness vs. loop vs. graph" split from Stage 7/8 made concrete —
# retrieve/grade/generate are deterministic-ish steps, not agent reasoning
# loops, so they're plain functions, not LLM calls wrapped in more LLM calls.

def make_retrieve_node(vector_store: Chroma):
    def retrieve(state: RAGState) -> dict:
        chunks = vector_store.similarity_search(state["query"], k=RETRIEVE_K)
        print(f"  [retrieve] query={state['query']!r} -> {len(chunks)} chunk(s)", file=sys.stderr)
        return {"chunks": chunks}

    return retrieve


def make_grade_node(llm: ChatOllama):
    def grade(state: RAGState) -> dict:
        # The corrective part of CRAG: don't trust retrieval blindly. Ask the
        # model to judge each chunk against the ORIGINAL question (not the
        # possibly-rewritten query) before it's allowed to inform the answer.
        graded_relevant = []
        for chunk in state["chunks"]:
            verdict = llm.invoke(
                f"Question: {state['original_query']}\n\n"
                f"Retrieved text:\n{chunk.page_content}\n\n"
                "Does this text contain information that helps answer the "
                "question? Reply with exactly one word: yes or no."
            )
            is_relevant = "yes" in verdict.content.strip().lower()
            print(
                f"  [grade] {chunk.metadata['source']}: "
                f"{'relevant' if is_relevant else 'not relevant'}",
                file=sys.stderr,
            )
            if is_relevant:
                graded_relevant.append(chunk)
        return {"graded_relevant": graded_relevant}

    return grade


def route_after_grading(state: RAGState) -> str:
    """The conditional edge — this IS the corrective-RAG decision, expressed
    as graph routing instead of an if/else buried inside one big function."""
    if state["graded_relevant"]:
        return "generate"
    if state["retries"] < MAX_RETRIES:
        return "rewrite_query"
    return "refuse"


def make_rewrite_node(llm: ChatOllama):
    def rewrite_query(state: RAGState) -> dict:
        # A weak first retrieval often means the query's wording doesn't
        # match the corpus's wording — rewriting and retrying once is
        # cheap and often fixes a vocabulary mismatch, not a "no answer
        # exists" situation.
        rewritten = llm.invoke(
            f"Original question: {state['original_query']}\n\n"
            "Rewrite this as a different, more specific search query that "
            "might match a company filing or earnings report. Reply with "
            "ONLY the rewritten query, nothing else."
        )
        new_query = rewritten.content.strip()
        print(f"  [rewrite] {state['original_query']!r} -> {new_query!r}", file=sys.stderr)
        return {"query": new_query, "retries": state["retries"] + 1}

    return rewrite_query


def make_generate_node(llm: ChatOllama):
    def generate(state: RAGState) -> dict:
        context = "\n\n".join(
            f"[{c.metadata['source']}] {c.page_content}" for c in state["graded_relevant"]
        )
        response = llm.invoke(
            f"Context:\n{context}\n\n"
            f"Question: {state['original_query']}\n\n"
            "Answer using ONLY the context above. Cite the source file "
            "in brackets, e.g. [annual_report_fy2025.txt]."
        )
        return {"answer": response.content}

    return generate


def refuse(state: RAGState) -> dict:
    # This is the whole point of "corrective": an ungrounded guess is worse
    # than admitting the corpus doesn't have the answer. No LLM call here —
    # there's nothing relevant to reason over, so don't pretend otherwise.
    print("  [refuse] no relevant chunks after retry — refusing rather than guessing", file=sys.stderr)
    return {
        "answer": (
            "I don't have enough information in the available documents to "
            "answer this question confidently. Retrieved content did not "
            "appear relevant even after rewriting the query."
        )
    }


def build_graph(vector_store: Chroma, llm: ChatOllama):
    graph = StateGraph(RAGState)
    graph.add_node("retrieve", make_retrieve_node(vector_store))
    graph.add_node("grade", make_grade_node(llm))
    graph.add_node("rewrite_query", make_rewrite_node(llm))
    graph.add_node("generate", make_generate_node(llm))
    graph.add_node("refuse", refuse)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges(
        "grade",
        route_after_grading,
        {"generate": "generate", "rewrite_query": "rewrite_query", "refuse": "refuse"},
    )
    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("generate", END)
    graph.add_edge("refuse", END)

    return graph.compile()


def run_agent(question: str) -> str:
    print(f"  [model] local via Ollama: {LOCAL_MODEL} (chat), {EMBEDDING_MODEL} (embeddings)", file=sys.stderr)
    vector_store = _build_vector_store()
    llm = _build_llm()
    app = build_graph(vector_store, llm)

    result = app.invoke({
        "query": question,
        "original_query": question,
        "chunks": [],
        "graded_relevant": [],
        "retries": 0,
        "answer": "",
    })
    return result["answer"]


if __name__ == "__main__":
    task = " ".join(sys.argv[1:]) or "What was Northwind Traders' revenue in fiscal 2025?"
    print(run_agent(task))
