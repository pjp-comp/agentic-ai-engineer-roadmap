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

HOW THE VECTOR STORE ACTUALLY WORKS, mechanically:

  Embedding: each document's TEXT gets converted into a VECTOR — a fixed-
  length list of floating-point numbers (768 of them, for nomic-embed-text)
  that represents that text's MEANING as one point in a high-dimensional
  space. This isn't a hash and it isn't a keyword index — two sentences
  that share zero words but mean similar things ("the CEO resigned" /
  "the chief executive stepped down") land as two NEARBY points in that
  768-dimensional space, because the embedding model was trained so that
  distance in vector-space tracks semantic similarity, not word overlap.

  Storing: Chroma (this example's vector database) keeps each vector next
  to the original text and its metadata (here: {"source": "..."}) in a
  structure built for fast nearest-neighbor lookup — given a NEW vector,
  quickly find which stored vectors are closest to it, without comparing
  against every single one by brute force once a collection gets large.
  (At 5 documents brute force would be instant either way — this matters
  at thousands-to-millions of vectors, which is the scale ChromaDB's
  underlying index structure is actually built for.)

  Retrieving: a QUESTION gets embedded with the exact same model, into a
  vector in the exact same space. "Nearest neighbors" to that vector are
  the documents whose MEANING is closest to the question's meaning — this
  is what similarity_search() is doing under the hood. Distance is
  typically cosine similarity (the angle between two vectors) rather than
  raw Euclidean distance, which is why "how alike is the MEANING" survives
  even when the phrasing is completely different. See retrieve()/grade()
  below for what happens with those nearest-neighbor results once found —
  vector search picks CANDIDATES, it doesn't decide they're actually
  relevant (that's grade()'s job, the corrective half of CRAG).

  The embedding model and the vector database are two SEPARATE pieces of
  software doing two different jobs: OllamaEmbeddings (below) does the
  text -> vector conversion; Chroma does the storing + nearest-neighbor
  search over vectors it doesn't generate itself. Swapping either one
  independently (a different embedding model, a different vector DB like
  pgvector or Pinecone) is a real, common thing to do, precisely because
  they're decoupled.

The vector store is PERSISTED to disk in this example's own folder
(.chroma_crag/), the same pattern stage04-longterm-memory-vectorstore
uses. Embedding is a real, separate step from querying — this repo splits
them into two commands: run build_index.py ONCE to embed and persist
_DOCS, then run agent.py as many times as you want against that already-
built index, with no re-embedding on any of those later runs. (agent.py
still auto-builds the index itself if you skip that step and .chroma_crag/
doesn't exist yet — a convenience fallback, not the intended workflow;
see build_index.py's own docstring for the full explanation of what
"building an index" means step by step.)

Usage:
    ollama pull llama3.2:3b        # one-time, ~2GB, shared with other examples
    ollama pull nomic-embed-text   # one-time, ~274MB, shared with stage04-longterm-memory-vectorstore
    uv run build_index.py                                      # build the index once, explicitly
    uv run agent.py "Why did the monkey refuse to give the crocodile his heart?"
    uv run agent.py "What is the capital of France?"           # forces the refusal path — nothing in the corpus is relevant
    uv run build_index.py --force                              # re-embed after editing _DOCS
"""

import shutil
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
PERSIST_DIR = Path(__file__).parent / ".chroma_crag"

# --- The corpus: a small set of documents the model did NOT train on -------
# A classic Panchatantra tale — "The Monkey and the Crocodile" — split into
# distinct passages (the story itself, its moral, background on the
# Panchatantra collection, and an unrelated second tale) rather than one
# fictional company's filings. Still deliberately structured so a correct
# answer requires retrieval AND discrimination: the "unrelated tale" doc
# shares characters/setting with the real story but answers a DIFFERENT
# question, so grading has real work to do, not just "found something or not."

_DOCS = [
    Document(
        page_content=(
            "The Monkey and the Crocodile (Panchatantra): A monkey named "
            "Raktamukha lived in a rose-apple tree by a river and befriended a "
            "crocodile who came to eat the sweet fruit. The monkey shared fruit "
            "with him daily. The crocodile's wife grew jealous and demanded the "
            "monkey's heart, believing an animal who ate such sweet fruit daily "
            "must have a sweet heart worth eating. The crocodile reluctantly "
            "agreed to trick his friend."
        ),
        metadata={"source": "monkey_and_crocodile_part1.txt"},
    ),
    Document(
        page_content=(
            "The Monkey and the Crocodile, continued: The crocodile invited the "
            "monkey to his home across the river, carrying him on his back. "
            "Midway, the crocodile admitted the plan to kill him for his heart. "
            "Thinking quickly, the monkey said he had left his heart behind in "
            "the rose-apple tree, as monkeys always do, and offered to fetch it. "
            "The crocodile swam back to shore, and the monkey leapt to safety "
            "into the tree, refusing to ever trust the crocodile again."
        ),
        metadata={"source": "monkey_and_crocodile_part2.txt"},
    ),
    Document(
        page_content=(
            "Moral of the Monkey and the Crocodile: The story teaches that "
            "presence of mind and wit can overcome betrayal by someone more "
            "powerful, and that true friendship should not be sacrificed to "
            "satisfy another's greed. It is one of the best-known tales from "
            "the Panchatantra's first book, Mitra-bheda (The Loss of Friends)."
        ),
        metadata={"source": "monkey_and_crocodile_moral.txt"},
    ),
    Document(
        page_content=(
            "About the Panchatantra: an ancient Indian collection of "
            "interrelated animal fables in Sanskrit verse and prose, compiled "
            "by Vishnu Sharma around the 3rd century BCE to teach principles of "
            "statecraft and wise conduct to young princes. It is organized into "
            "five books (tantras), each built around a frame story containing "
            "further nested tales."
        ),
        metadata={"source": "panchatantra_background.txt"},
    ),
    Document(
        page_content=(
            "The Tortoise and the Geese (a different Panchatantra tale): a "
            "talkative tortoise asked two geese friends to carry him during a "
            "drought by gripping a stick in his mouth while they held the ends. "
            "He was warned not to speak mid-flight. When onlookers below "
            "mocked the sight, the tortoise opened his mouth to retort, lost "
            "his grip, and fell to his death — a warning about the dangers of "
            "talking too much at the wrong moment."
        ),
        metadata={"source": "tortoise_and_geese.txt"},
    ),
]


class RAGState(TypedDict):
    query: str
    original_query: str
    chunks: list[Document]
    graded_relevant: list[Document]
    retries: int
    answer: str


def _build_vector_store(rebuild: bool = False) -> Chroma:
    """Opens the persisted index at PERSIST_DIR. The intended workflow is
    running build_index.py once, BEFORE this ever runs, so this function
    normally only ever OPENS an existing index — it doesn't embed
    anything.

    The is_new branch below is a fallback for skipping that step: if
    PERSIST_DIR doesn't exist yet, this embeds _DOCS itself rather than
    failing outright, so `uv run agent.py` alone still works standalone.
    That fallback is what makes this function look like it "builds" the
    store — really, in the intended workflow, build_index.py already did
    that, and this just opens what's on disk.

    rebuild=True wipes PERSIST_DIR first, forcing a fresh embed here
    directly, as a shortcut to `uv run build_index.py --force` — the
    escape hatch for "I edited _DOCS and need the index to reflect that"
    (a persisted index doesn't know its source documents changed
    underneath it; nothing here diffs old vs. new automatically).
    """
    if rebuild and PERSIST_DIR.exists():
        shutil.rmtree(PERSIST_DIR)

    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    is_new = not PERSIST_DIR.exists()
    store = Chroma(
        collection_name="crag_demo",
        embedding_function=embeddings,
        persist_directory=str(PERSIST_DIR),
    )
    if is_new:
        print(
            f"  [vector store] WARNING: no index found at {PERSIST_DIR.name}/ -- "
            f"embedding {len(_DOCS)} document(s) now as a fallback. "
            "Run `uv run build_index.py` first next time to skip this.",
            file=sys.stderr,
        )
        store.add_documents(_DOCS)
    else:
        print(f"  [vector store] opened persisted index at {PERSIST_DIR.name}/ (no embedding needed)", file=sys.stderr)
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
        # similarity_search() does two things in one call: embeds
        # state["query"] into a vector using the SAME embedding model
        # that built the index (EMBEDDING_MODEL — mismatching models here
        # would put the query in a different vector space than the
        # documents, making distance meaningless), then asks Chroma for
        # the k stored vectors nearest to it. "Nearest" is cosine
        # similarity by default — closest in MEANING, not closest in
        # literal wording. This returns k CANDIDATES; it does not judge
        # whether they actually answer the question — that's grade()'s job.
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
                f"Retrieved text:\n{chunk.page_content}\n\n"
                f"Question: {state['original_query']}\n\n"
                "Is the retrieved text relevant to answering this question? "
                "Answer yes if the text discusses the same topic, characters, or events "
                "as the question, even if it doesn't state the answer outright. "
                "Answer no only if the text is about something unrelated. "
                "Reply with exactly one word: yes or no."
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
            "might match a passage from a folktale or story. Reply with "
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
            "in brackets, e.g. [monkey_and_crocodile_part1.txt]."
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


def run_agent(question: str, rebuild: bool = False) -> str:
    print(f"  [model] local via Ollama: {LOCAL_MODEL} (chat), {EMBEDDING_MODEL} (embeddings)", file=sys.stderr)
    vector_store = _build_vector_store(rebuild=rebuild)
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
    argv = sys.argv[1:]
    rebuild = "--rebuild" in argv
    if rebuild:
        argv = [a for a in argv if a != "--rebuild"]
    task = " ".join(argv) or "Why did the monkey refuse to give the crocodile his heart?"
    print(run_agent(task, rebuild=rebuild))
