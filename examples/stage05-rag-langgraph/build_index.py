"""
The explicit "build the index once" command — run this BEFORE agent.py,
the same way a real RAG pipeline separates an offline indexing job from
the online query service that reads from it. agent.py still auto-builds
the index on its own first run as a fallback (so `uv run agent.py` alone
still works, matching every other example in this repo), but that's a
convenience for forgetting this step, not the intended workflow.

What "building the index" actually means, mechanically:

  1. Each Document in _DOCS gets sent to the embedding model
     (nomic-embed-text, via Ollama) one at a time. The model reads the
     text and returns a vector -- a fixed-length list of floating-point
     numbers (768 of them, for nomic-embed-text) that represents the
     text's MEANING as a point in a high-dimensional space. Two pieces
     of text with similar meaning end up as two points that are close
     together in that space; unrelated text ends up far apart. This is
     the entire trick vector search is built on -- see agent.py's module
     docstring and this repo's README for the fuller explanation.

  2. Chroma stores each vector alongside the original text and its
     metadata (here, just {"source": "..."}) in a local SQLite-backed
     directory (.chroma_crag/). This is the "index" -- a structure built
     specifically so that, later, a NEW vector (the query's embedding)
     can be compared against all the stored ones quickly, without
     re-reading or re-embedding the original documents.

  3. Nothing is compared or searched yet at this point -- indexing and
     querying are two separate operations. This script only does step 1
     and 2. agent.py's retrieve() node is what does the comparison, on
     each question, against whatever's already sitting in .chroma_crag/.

Usage:
    ollama pull nomic-embed-text   # one-time, ~274MB, shared with stage04-longterm-memory-vectorstore
    uv run build_index.py          # build fresh, or no-op if already built
    uv run build_index.py --force  # wipe and rebuild even if one exists
                                    #   (use after editing _DOCS in agent.py)
"""

import shutil
import sys

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from pathlib import Path

from agent import _DOCS, EMBEDDING_MODEL, PERSIST_DIR

load_dotenv(Path(__file__).resolve().parents[2] / ".env")


def build_index(force: bool = False) -> None:
    if PERSIST_DIR.exists():
        if not force:
            print(f"Index already exists at {PERSIST_DIR.name}/ — nothing to do.")
            print("Use --force to wipe and rebuild (e.g. after editing _DOCS).")
            return
        print(f"--force: removing existing index at {PERSIST_DIR.name}/")
        shutil.rmtree(PERSIST_DIR)

    print(f"Embedding {len(_DOCS)} document(s) with {EMBEDDING_MODEL} via Ollama...")
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    store = Chroma(
        collection_name="crag_demo",
        embedding_function=embeddings,
        persist_directory=str(PERSIST_DIR),
    )
    store.add_documents(_DOCS)
    print(f"Done. Index written to {PERSIST_DIR.name}/ — {len(_DOCS)} document(s), "
          f"ready for agent.py to query without re-embedding.")


if __name__ == "__main__":
    build_index(force="--force" in sys.argv[1:])
