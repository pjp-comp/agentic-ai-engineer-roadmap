"""
Multi-vector indexing -- the SECOND half of this example's retrieval
technique (ingest.py's Elements are the first half). This is the current
best-practice pattern for making tables and images in a PDF actually
retrievable by MEANING, not just present in the index:

  For each Element:
    1. Generate a short TEXT SUMMARY of it.
         - text elements: the text IS already the summary -- no model
           call needed, embed it directly.
         - table elements: ask the chat model (LOCAL_MODEL) to describe
           what the table shows in a sentence or two. A table's raw
           "Region | 2023 | 2024 | YoY" grid embeds poorly -- vector
           search over raw table text tends to match on incidental
           formatting/number patterns rather than the table's actual
           subject. A generated summary ("adoption rates by region,
           showing East Asia's 7.7pp year-over-year gain...") embeds
           the way a normal QUESTION about that table would be phrased,
           which is what actually makes retrieval accurate.
         - image elements: ask the VISION model (VISION_MODEL) what the
           image depicts. This is the only element type where this step
           is not optional -- raw image bytes have no text representation
           at all; without a real vision-model description, an image is
           entirely invisible to a text-embedding-based vector search.
    2. Embed the SUMMARY (not the raw content) with EMBEDDING_MODEL.
    3. Store the summary's vector in Chroma, but keep the ORIGINAL raw
       content (full table text, or a reference to the image) in a
       separate docstore, linked by element_id.

This is why it's called MULTI-vector / "parent document" retrieval: the
vector that gets matched during search (the summary) is deliberately a
DIFFERENT piece of text from what gets handed to the final answer-writing
model (the raw table or the image's full description). Retrieval quality
comes from the summary; answer accuracy comes from the original content
having been preserved untouched rather than lossily embedded directly.
This is the same pattern LangChain's MultiVectorRetriever and
Unstructured.io's RAG reference architecture both use for exactly this
table/image problem.
"""

import base64
import json
import sys
from pathlib import Path

import ollama
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

from ingest import Element

LOCAL_MODEL = "llama3.1:8b"
VISION_MODEL = "qwen2.5vl:7b"
EMBEDDING_MODEL = "nomic-embed-text"


def _summarize_table(chat_model: str, element: Element) -> str:
    resp = ollama.chat(
        model=chat_model,
        messages=[{
            "role": "user",
            "content": (
                "Summarize what this table shows in 2-3 sentences. Mention the "
                "column headers and any notable values or trends. Be specific "
                "with numbers where relevant.\n\n"
                f"Table:\n{element.content}"
            ),
        }],
        think=False,
    )
    return resp.message.content.strip()


def _summarize_image(vision_model: str, element: Element) -> str:
    """The one step in this whole pipeline that REQUIRES a vision-capable
    model -- no text model can do this, because the input isn't text.

    VISION_MODEL history, kept here because it's a real, verified finding
    about model choice mattering, not just a config value: this pipeline
    originally used llava:7b, which was NOT reliable at reading discrete
    values off a multi-bar/multi-category chart -- asked plainly to just
    list a 5-category bar chart's axis labels top to bottom, it invented a
    16-item list mixing real and fabricated category names, and tightening
    the prompt to demand every value made fabrication WORSE (more
    confident invention to fill out the list). Swapping to qwen2.5vl:7b
    fixed this outright -- the same isolated "list categories and values"
    test that broke llava returned all 5 real categories with all 5 real
    values, correctly, on the first try. This was a genuine model-capability
    gap (structured chart/plot reading), not a prompting problem -- no
    amount of instruction tuning fixed it on llava, and almost no
    instruction tuning was needed once the model itself was strong enough.

    Caption grounding (element.caption, captured in ingest.py as text
    within a tight vertical window of the image's own bbox, not the whole
    page) is kept regardless of which vision model is used -- it's still
    a real, independent improvement: when the surrounding prose states a
    chart's headline number in words ("Installation added 680,000
    jobs..."), grounding lets the model corroborate that reading rather
    than relying on pixel-reading alone for numbers that are ALSO stated
    in text nearby.
    """
    b64 = base64.b64encode(element.image_bytes).decode("ascii")
    caption_block = (
        f"\n\nText near this image on the page (for context -- if this text states "
        f"specific numbers or facts about what's shown in the image, treat that as "
        f"confirmation of your own reading):\n{element.caption}"
        if element.caption else ""
    )
    resp = ollama.chat(
        model=vision_model,
        messages=[{
            "role": "user",
            "content": (
                "Describe what this image shows. If it's a chart or graph, "
                "state the title, both axes, what's being compared, and "
                "EVERY category/data point with its exact value -- not just "
                "the largest or most notable one. Be specific with numbers."
                f"{caption_block}"
            ),
            "images": [b64],
        }],
        think=False,
    )
    return resp.message.content.strip()


def summarize_elements(elements: list[Element]) -> list[dict]:
    """Returns a list of {element_id, kind, page, source, summary, raw}
    dicts -- 'raw' is the original content (table text, or a placeholder
    for images since we don't re-store binary image bytes here) that
    gets handed to generate() later, NOT what gets embedded.
    """
    summarized = []
    for el in elements:
        if el.kind == "text":
            summary = el.content   # already text -- no model call needed
            raw = el.content
        elif el.kind == "table":
            print(f"  [summarize] {el.element_id} (page {el.page}): asking {LOCAL_MODEL} to describe the table", file=sys.stderr)
            summary = _summarize_table(LOCAL_MODEL, el)
            raw = el.content   # the FULL table text, preserved for the final answer
        elif el.kind == "image":
            print(f"  [summarize] {el.element_id} (page {el.page}): asking {VISION_MODEL} to describe the image", file=sys.stderr)
            summary = _summarize_image(VISION_MODEL, el)
            raw = summary   # for images, the generated description IS the
                             # only text representation we have -- there's
                             # no "more complete" raw text form to fall back to
        else:
            continue

        summarized.append({
            "element_id": el.element_id,
            "kind": el.kind,
            "page": el.page,
            "source": el.source,
            "summary": summary,
            "raw": raw,
        })
    return summarized


def build_multi_vector_index(elements: list[Element], persist_dir: Path) -> tuple[Chroma, dict]:
    """Returns (vector_store, docstore). vector_store holds ONE vector per
    element -- the embedding of its SUMMARY, TAGGED with which source PDF
    it came from. This is a SHARED index across every PDF ever indexed
    into persist_dir, not one index per document: adding a second PDF
    means calling this again with that PDF's elements, and both PDFs'
    content becomes searchable together in the same vector_store,
    distinguished by the "source" metadata field on each vector rather
    than by which directory it lives in.

    docstore.json is correspondingly MERGED, not overwritten -- existing
    entries from previously-indexed PDFs stay untouched; only this call's
    elements get added or updated. element_id is namespaced by source
    (e.g. "sample.pdf::table-8") specifically so two different PDFs'
    "table-8" can never collide and silently overwrite each other here.
    """
    summarized = summarize_elements(elements)

    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vector_store = Chroma(
        collection_name="multimodal_rag",
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
    )

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

    print(f"  [index] embedded {len(docs)} summarized element(s), docstore now has {len(docstore)} total ({docstore_path.name})", file=sys.stderr)
    return vector_store, docstore


def load_docstore(persist_dir: Path) -> dict:
    docstore_path = persist_dir / "docstore.json"
    if not docstore_path.exists():
        return {}
    return json.loads(docstore_path.read_text())
