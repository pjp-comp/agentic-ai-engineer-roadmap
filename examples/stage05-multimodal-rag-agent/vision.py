"""
VISION / SUMMARIZATION -- turns each chunking.Element into a short TEXT
SUMMARY, the piece that actually gets embedded (see embeddings.py) and
searched (see retrieval.py). This is the core of the "multi-vector
retrieval" technique this example builds:

  - text elements: the text IS already the summary -- no model call
    needed, embed it directly.
  - table elements: ask the chat model (LOCAL_MODEL) to describe what
    the table shows in a sentence or two. A table's raw
    "Region | 2023 | 2024 | YoY" grid embeds poorly -- vector search
    over raw table text tends to match on incidental formatting/number
    patterns rather than the table's actual subject. A generated
    summary ("adoption rates by region, showing East Asia's 7.7pp
    year-over-year gain...") embeds the way a normal QUESTION about
    that table would be phrased, which is what actually makes retrieval
    accurate.
  - image elements: ask the VISION model (VISION_MODEL) what the image
    depicts. This is the only element type where this step is not
    optional -- raw image bytes have no text representation at all;
    without a real vision-model description, an image is entirely
    invisible to a text-embedding-based vector search.

This is why it's called MULTI-vector / "parent document" retrieval: the
vector that gets matched during search (the summary, built here) is
deliberately a DIFFERENT piece of text from what gets handed to the
final answer-writing model (the raw table or the image's full
description, preserved in vector_store.py's docstore). Retrieval quality
comes from the summary; answer accuracy comes from the original content
having been preserved untouched rather than lossily embedded directly.
This is the same pattern LangChain's MultiVectorRetriever and
Unstructured.io's RAG reference architecture both use for exactly this
table/image problem.
"""

import base64
import hashlib
import sys

import ollama

from chunking import Element

LOCAL_MODEL = "llama3.1:8b"
VISION_MODEL = "qwen2.5vl:7b"


def summarize_table(element: Element, chat_model: str = LOCAL_MODEL) -> str:
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


def summarize_image(element: Element, vision_model: str = VISION_MODEL) -> str:
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
    See embeddings.py's module docstring for why CLIP is NOT a substitute
    for this step -- CLIP has no capability to read a chart's values at all.

    Caption grounding (element.caption, captured in chunking.py as text
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
    dicts -- 'raw' is the original content (table text, or the vision
    model's own description for images, since we don't re-store binary
    image bytes here) that gets handed to generation.py's generate()
    later, NOT what gets embedded.

    Image dedup: a real, verified inefficiency this caught -- a
    multi-page PDF with a repeated letterhead/logo in the header of
    EVERY page produces one image Element per page, all byte-identical.
    Without dedup, a 20-page filing calls the vision model 20 times to
    describe the exact same logo, each call taking real wall-clock time
    for zero new information. The fix: hash each image's raw bytes
    (SHA-256, cheap and exact -- a repeated logo is a byte-for-byte
    identical PNG/JPEG, not merely visually similar, so a hash match is
    a genuine duplicate, not a heuristic guess) and call
    summarize_image() only for the FIRST occurrence of each unique hash.
    Every later element with the same hash reuses that description
    directly -- still gets its own element_id, its own entry in the
    vector store and docstore (so retrieval/citation by page still work
    correctly per-occurrence), just without paying for a redundant model
    call to re-describe pixels already described once.
    """
    image_summary_by_hash: dict[str, str] = {}
    summarized = []

    for el in elements:
        if el.kind == "text":
            summary = el.content   # already text -- no model call needed
            raw = el.content
        elif el.kind == "table":
            print(f"  [vision] {el.element_id} (page {el.page}): asking {LOCAL_MODEL} to describe the table", file=sys.stderr)
            summary = summarize_table(el)
            print(f"  [vision] {el.element_id} description: {summary}", file=sys.stderr)
            raw = el.content   # the FULL table text, preserved for the final answer
        elif el.kind == "image":
            image_hash = hashlib.sha256(el.image_bytes).hexdigest()
            if image_hash in image_summary_by_hash:
                summary = image_summary_by_hash[image_hash]
                print(f"  [vision] {el.element_id} (page {el.page}): duplicate image (matches an earlier one) -- reusing its description, no model call", file=sys.stderr)
            else:
                print(f"  [vision] {el.element_id} (page {el.page}): asking {VISION_MODEL} to describe the image", file=sys.stderr)
                summary = summarize_image(el)
                print(f"  [vision] {el.element_id} description: {summary}", file=sys.stderr)
                image_summary_by_hash[image_hash] = summary
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
