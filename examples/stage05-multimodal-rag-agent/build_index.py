"""
The "index everything in assets/" command -- run this before agent.py.
Same split as stage05-rag-langgraph's build_index.py: indexing (this
file) and querying (agent.py) are two separate concerns, run at
different times, for different reasons -- a real production RAG system
re-indexes on a schedule or on document change, and serves queries
continuously; conflating the two into "the first query happens to also
build the index" hides that real difference.

Source files live in assets/ -- this example does NOT ship a PDF
generator or any sample files. Drop your own file(s) into assets/ and
run this with no arguments; it scans the directory and indexes
everything it finds, of any SUPPORTED type:

  .pdf                        -> full element-aware extraction (text/table/image)
  .txt, .md                   -> the file's own text, paragraph-chunked
  .png, .jpg, .jpeg, .webp    -> the whole file as one image, captioned by
                                 the vision model (same as an image found
                                 INSIDE a PDF, just with no surrounding
                                 page text to draw a caption from)

An unrecognized extension (e.g. .docx, .csv) is a loud error at index
time, not a silent skip -- see chunking.parse_file()'s docstring.

ONE SHARED INDEX, not one index per file. Every file you index gets added
into the SAME .chroma_multimodal/ directory and the SAME docstore.json --
each file's content becomes searchable ALONGSIDE whatever was already
indexed, not in a separate, isolated store. This is what makes "just add
more files later" actually work: the retrieval step in agent.py searches
across every indexed document at once by default, the same way a real
RAG system over a growing document collection would, distinguishing
WHICH document an answer came from via the "source" field on each
element (see chunking.py's Element.source) rather than via which
directory got queried.

Pipeline stages, each in its own file (see that file's docstring for the
full mechanics of each stage):
  chunking.py      -- parse_file(): dispatches by extension to
                       parse_pdf() / parse_text_file() / parse_image_file()
  vision.py         -- summarize_elements(): Elements -> text summaries
                       (table/image summaries need a model call each)
  embeddings.py     -- which embedding model turns summaries into vectors
  vector_store.py   -- owns the Chroma collection + docstore.json,
                       build_multi_vector_index() ties the above together

Usage:
    ollama pull llama3.1:8b        # one-time, ~4.9GB
    ollama pull qwen2.5vl:7b        # one-time, ~6GB, vision model
    ollama pull nomic-embed-text    # one-time, ~274MB
    # drop your file(s) into assets/ (PDFs, .txt/.md, or images), then
    # ONE command indexes all of them:
    uv run build_index.py
        -> scans assets/ for every supported file, indexes whatever
           isn't already indexed, skips the rest. Add a 5th file later
           and re-run this same command -- only the new one gets processed.
    uv run build_index.py --force
        -> wipes the shared index completely, then rebuilds it from
           EVERY supported file currently in assets/ (not just one)
    uv run build_index.py --file your_file.pdf [--force]
        -> target a single file instead of the whole assets/ directory,
           for when you only want to (re)index one. --pdf also works,
           as an alias, for anyone used to the older flag name.
    uv run build_index.py --describe-images
        -> print every INDEXED image's source file, page, and the full
           vision-model description it was given -- what actually got
           embedded and would be used to answer a question, for every
           image in the index, in one place. Read this if a chart-based
           answer looks wrong: the description here IS the summary
           retrieval searches over (see vision.py/vector_store.py).
    uv run build_index.py --describe-images --source your_file.pdf
        -> narrow the above to just one indexed file's images
"""

import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv

from chunking import IMAGE_EXTENSIONS, TEXT_EXTENSIONS, parse_file
from vector_store import PERSIST_DIR, build_multi_vector_index, load_docstore, sources_already_indexed

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

ASSETS_DIR = Path(__file__).parent / "assets"
SUPPORTED_EXTENSIONS = {".pdf"} | TEXT_EXTENSIONS | IMAGE_EXTENSIONS


def _index_one(path: Path, already_indexed: set[str]) -> None:
    if path.name in already_indexed:
        print(f"{path.name}: already indexed, skipping (use --force to rebuild everything)")
        return

    print(f"{path.name}: parsing...")
    elements = parse_file(path)
    by_kind: dict[str, int] = {}
    for el in elements:
        by_kind[el.kind] = by_kind.get(el.kind, 0) + 1
    print(f"  {', '.join(f'{v} {k}' for k, v in by_kind.items())}")

    print(f"  Summarizing + embedding (tables/images need a model call each, this takes a minute)...")
    build_multi_vector_index(elements, PERSIST_DIR)


def build_index(paths: list[Path], force: bool = False) -> None:
    """paths is the full set of files to ensure are indexed -- for the
    no-argument / assets/-scan invocation this is EVERY supported file
    in assets/, so a single command indexes an arbitrary number of
    files, not one per command. Already-indexed files are skipped
    automatically (checked against the shared docstore, not re-parsed or
    re-embedded) unless --force wipes the index first.
    """
    missing = [p for p in paths if not p.exists()]
    for p in missing:
        print(f"No {p.name} found in {ASSETS_DIR.name}/ -- skipping.")
    paths = [p for p in paths if p.exists()]
    if not paths:
        print(f"Nothing to index. Drop file(s) into {ASSETS_DIR.name}/ first.")
        return

    if force and PERSIST_DIR.exists():
        print(f"--force: wiping the shared index at {PERSIST_DIR.name}/ "
              f"and rebuilding from {len(paths)} file(s) in {ASSETS_DIR.name}/")
        shutil.rmtree(PERSIST_DIR)

    PERSIST_DIR.mkdir(exist_ok=True)

    already_indexed = set() if force else sources_already_indexed()
    for path in paths:
        _index_one(path, already_indexed)

    sources = sources_already_indexed()
    print(f"\nDone. {PERSIST_DIR.name}/ now contains {len(sources)} document(s): {', '.join(sorted(sources))}")


def describe_images(source_filter: str | None = None) -> None:
    """Prints every INDEXED image element's source, page, and its full
    vision-model description (the 'summary' field -- for images, this IS
    the ONLY text representation of the image that exists anywhere in the
    pipeline; see vision.py's summarize_image() and vector_store.py's
    docstring for why). This is exactly what retrieval.retrieve() embeds
    and searches, and exactly what a question about that image would be
    answered from -- reading it here is the fastest way to see WHICH
    image produced WHAT description, without re-running a query and
    working backward from an answer.
    """
    docstore = load_docstore(PERSIST_DIR)
    if not docstore:
        print("No index found. Run `uv run build_index.py` first.")
        return

    images = [
        rec for rec in docstore.values()
        if rec["kind"] == "image" and (source_filter is None or rec["source"] == source_filter)
    ]
    if not images:
        scope = f" for source={source_filter!r}" if source_filter else ""
        print(f"No indexed images found{scope}.")
        return

    images.sort(key=lambda r: (r["source"], r["page"]))
    print(f"{len(images)} indexed image(s):\n")
    for rec in images:
        print(f"=== {rec['source']}, page {rec['page']} ({rec['element_id']}) ===")
        print(rec["summary"])
        print()


if __name__ == "__main__":
    argv = sys.argv[1:]
    force = "--force" in argv
    if force:
        argv = [a for a in argv if a != "--force"]

    source_filter = None
    if "--source" in argv:
        idx = argv.index("--source")
        source_filter = argv[idx + 1]
        argv = argv[:idx] + argv[idx + 2:]

    if "--describe-images" in argv:
        describe_images(source_filter=source_filter)
        sys.exit(0)

    target_flag = "--file" if "--file" in argv else ("--pdf" if "--pdf" in argv else None)
    if target_flag:
        idx = argv.index(target_flag)
        paths = [ASSETS_DIR / argv[idx + 1]]
    else:
        paths = sorted(
            p for p in (ASSETS_DIR.glob("*") if ASSETS_DIR.exists() else [])
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        if not paths:
            print(f"No supported files found in {ASSETS_DIR.name}/. "
                  f"Drop a .pdf, .txt/.md, or image ({', '.join(sorted(IMAGE_EXTENSIONS))}) there first.")
            sys.exit(1)

    build_index(paths=paths, force=force)
