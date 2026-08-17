"""
VECTOR STORE -- owns the Chroma collection and the docstore, the two
pieces of on-disk state this pipeline builds and reads. Isolated from
vision.py (which decides WHAT to embed) and retrieval.py (which decides
HOW to search) so "where the vectors live" is a separate concern from
both.

ONE SHARED INDEX, not one per PDF. Every PDF you index goes into the SAME
.chroma_multimodal/ directory and the SAME docstore.json -- indexing a
second PDF adds its content ALONGSIDE whatever's already there, not into
a separate, isolated store. This is what makes "just add more PDFs
later" actually work: retrieval.py searches across every indexed
document at once by default, distinguishing WHICH document an answer
came from via the "source" field on each element (see chunking.py's
Element.source) rather than via which directory got queried.

The docstore split -- WHY two stores, not one:
  Chroma (.chroma_multimodal/)  -- holds ONE VECTOR per element: the
                                    embedding of its SUMMARY (see
                                    vision.py). This is what
                                    similarity_search() searches over.
  docstore.json                 -- holds the FULL RECORD per element,
                                    including its RAW content (the whole
                                    table, or the image's full
                                    description) -- what
                                    generation.py's generate() actually
                                    reads to build an answer. Looked up
                                    by element_id, keyed off what
                                    Chroma's metadata returns.
A vector store answers "which elements are relevant"; the docstore
answers "what do those elements actually say." Conflating the two --
embedding and answering from the same text -- is the naive-RAG mistake
this whole pipeline is built to avoid.
"""

import json
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document

from embeddings import build_text_embeddings
from vision import summarize_elements
from chunking import Element

COLLECTION_NAME = "multimodal_rag"
PERSIST_DIR = Path(__file__).parent / ".chroma_multimodal"


def open_vector_store(persist_dir: Path = PERSIST_DIR) -> Chroma:
    """Opens (or creates, if empty) the shared Chroma collection. Used
    both for indexing (vector_store.build_index) and for querying
    (retrieval.py) -- same collection, same embedding function, so a
    query embedded here lands in the same space as every summary that
    was embedded during indexing.
    """
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=build_text_embeddings(),
        persist_directory=str(persist_dir),
    )


def build_multi_vector_index(elements: list[Element], persist_dir: Path = PERSIST_DIR) -> tuple[Chroma, dict]:
    """Returns (vector_store, docstore). vector_store holds ONE vector per
    element -- the embedding of its SUMMARY, TAGGED with which source PDF
    it came from. docstore.json is MERGED, not overwritten -- existing
    entries from previously-indexed PDFs stay untouched; only this call's
    elements get added or updated. element_id is namespaced by source
    (e.g. "sample.pdf::table-8") specifically so two different PDFs'
    "table-8" can never collide and silently overwrite each other here.
    """
    summarized = summarize_elements(elements)

    vector_store = open_vector_store(persist_dir)

    docs = [
        Document(
            page_content=item["summary"],
            metadata={
                "element_id": item["element_id"], "kind": item["kind"],
                "page": item["page"], "source": item["source"],
            },
        )
        for item in summarized
    ]
    vector_store.add_documents(docs)

    docstore = load_docstore(persist_dir)   # merge into whatever's already indexed
    docstore.update({item["element_id"]: item for item in summarized})
    docstore_path = persist_dir / "docstore.json"
    docstore_path.write_text(json.dumps(docstore, indent=2))

    print(f"  [vector_store] embedded {len(docs)} summarized element(s), docstore now has {len(docstore)} total ({docstore_path.name})")
    return vector_store, docstore


def load_docstore(persist_dir: Path = PERSIST_DIR) -> dict:
    docstore_path = persist_dir / "docstore.json"
    if not docstore_path.exists():
        return {}
    return json.loads(docstore_path.read_text())


def sources_already_indexed(persist_dir: Path = PERSIST_DIR) -> set[str]:
    """Which PDF filenames already have elements in the shared docstore --
    used both to skip redundant re-indexing and to show what's in the
    store so far.
    """
    docstore = load_docstore(persist_dir)
    return {record.get("source", "") for record in docstore.values() if record.get("source")}
