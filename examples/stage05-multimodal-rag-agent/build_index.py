"""
The explicit "parse a PDF and add it to the index" command -- run this
ONCE per PDF, before agent.py. Same split as stage05-rag-langgraph's
build_index.py: indexing (this file) and querying (agent.py) are two
separate concerns, run at different times, for different reasons -- a
real production RAG system re-indexes on a schedule or on document
change, and serves queries continuously; conflating the two into "the
first query happens to also build the index" hides that real difference.

ONE SHARED INDEX, not one index per PDF. Every PDF you index gets added
into the SAME .chroma_multimodal/ directory and the SAME docstore.json --
run this again with a different --pdf and its content becomes searchable
ALONGSIDE whatever was already indexed, not in a separate, isolated
store. This is what makes "just add more PDFs later" actually work: the
retrieval step in agent.py searches across every indexed document at
once by default, the same way a real RAG system over a growing document
collection would, distinguishing WHICH document an answer came from via
the "source" field on each element (see ingest.py's Element.source and
index.py's docstring) rather than via which directory got queried.

What this actually does, step by step (see ingest.py and index.py for
the full mechanics of each step):

  1. ingest.parse_pdf(pdf) -- pymupdf reads every page, splits content
     into typed Elements: text blocks (read directly), tables (read via
     pymupdf's table-detection, as real structured text), and images
     (extracted as raw bytes, page-tagged) -- each element_id namespaced
     by the source filename so two PDFs' elements never collide.

  2. index.build_multi_vector_index(elements) -- for EACH element:
       - text elements: used as-is, no model call
       - table elements: llama3.1:8b generates a 2-3 sentence summary of
         what the table shows
       - image elements: llava:7b (a VISION model) generates a 2-3
         sentence description of what the image depicts -- this is the
         one step that requires a vision-capable model; no text model
         can look at image bytes
     Each summary gets embedded (nomic-embed-text) into the SHARED
     Chroma collection. The ORIGINAL raw content (full table text, or
     the image's own generated description) is MERGED into the shared
     docstore.json, linked by element_id, for agent.py to use at
     answer-generation time.

Two source documents ship with this example, both indexable into the
same store:
  sample.pdf          -- 3 pages, 1 table, 1 chart (the simple case)
  sample_complex.pdf  -- 5 pages, 3 tables, 3 charts, with topically
                          adjacent tables/charts (two different tables
                          both about "% growth," two different charts
                          both trending "over time") so retrieval and
                          grading have to discriminate WHICH one answers
                          a given question, not just detect "a table
                          exists somewhere."

Usage:
    ollama pull llama3.1:8b        # one-time, ~4.9GB
    ollama pull llava:7b            # one-time, ~4.7GB, vision model
    ollama pull nomic-embed-text    # one-time, ~274MB
    uv run build_index.py                           # indexes sample.pdf into the shared store
    uv run build_index.py --pdf sample_complex.pdf   # ADDS the complex doc alongside it
    uv run build_index.py --pdf your_own_file.pdf    # ADDS any PDF you drop in this directory
    uv run build_index.py --force                    # wipe the ENTIRE shared index and rebuild
                                                       # from just the given --pdf (or sample.pdf)
"""

import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv

from ingest import parse_pdf
from index import build_multi_vector_index

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

DEFAULT_PDF = Path(__file__).parent / "sample.pdf"
PERSIST_DIR = Path(__file__).parent / ".chroma_multimodal"


def build_index(pdf_path: Path = DEFAULT_PDF, force: bool = False) -> Path | None:
    if not pdf_path.exists():
        print(f"No {pdf_path.name} found. Generate it first (see generate_sample_pdf.py or "
              "generate_complex_pdf.py's docstring), or drop your own PDF at this path.")
        return None

    if force and PERSIST_DIR.exists():
        print(f"--force: wiping the ENTIRE shared index at {PERSIST_DIR.name}/ "
              "(this removes every previously-indexed PDF, not just this one)")
        shutil.rmtree(PERSIST_DIR)

    PERSIST_DIR.mkdir(exist_ok=True)

    already_indexed = _sources_already_indexed()
    if pdf_path.name in already_indexed and not force:
        print(f"{pdf_path.name} is already indexed in {PERSIST_DIR.name}/ — nothing to do.")
        print(f"Already-indexed source(s): {', '.join(sorted(already_indexed))}")
        print("Use --force to wipe the whole shared index and start over.")
        return PERSIST_DIR

    print(f"Parsing {pdf_path.name}...")
    elements = parse_pdf(pdf_path)
    by_kind = {}
    for el in elements:
        by_kind[el.kind] = by_kind.get(el.kind, 0) + 1
    print(f"  {', '.join(f'{v} {k}' for k, v in by_kind.items())}")

    print("Summarizing + embedding elements (tables/images need a model call each, this takes a minute)...")
    build_multi_vector_index(elements, PERSIST_DIR)

    sources = _sources_already_indexed()
    print(f"Done. {PERSIST_DIR.name}/ now contains {len(sources)} document(s): {', '.join(sorted(sources))}")
    return PERSIST_DIR


def _sources_already_indexed() -> set[str]:
    """Which PDF filenames already have elements in the shared docstore --
    used both to skip redundant re-indexing and to show what's in the
    store so far.
    """
    from index import load_docstore
    docstore = load_docstore(PERSIST_DIR)
    return {record.get("source", "") for record in docstore.values() if record.get("source")}


if __name__ == "__main__":
    argv = sys.argv[1:]
    pdf_path = DEFAULT_PDF
    if "--pdf" in argv:
        idx = argv.index("--pdf")
        pdf_path = Path(__file__).parent / argv[idx + 1]
    build_index(pdf_path=pdf_path, force="--force" in argv)
