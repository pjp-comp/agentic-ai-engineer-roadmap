"""
Element-aware PDF parsing -- turns sample.pdf into a list of typed
Elements (text blocks, tables, images), each tagged with the page it
came from. This is the FIRST half of the "multi-vector retrieval"
technique this example builds (see index.py for the second half:
generating a searchable summary per element).

Why element-aware, not "extract all text into one blob": a PDF's table
and a PDF's embedded chart are fundamentally different kinds of content
that need different extraction paths:

  - Text blocks   -> pymupdf reads them directly as real text. No model
                     call needed at all.
  - Tables        -> pymupdf's table-detection (page.find_tables()) reads
                     the actual grid structure -- rows, columns, cell
                     text -- as real text, because a genuine PDF table
                     (not a scanned image of one) IS text with layout
                     information, not a picture. This is fast, free
                     (no LLM call), and exact.
  - Images        -> pymupdf can extract the raw image bytes, but has NO
                     way to know what a chart or diagram DEPICTS -- "a
                     PNG of some pixels" isn't retrievable by meaning.
                     This is the one element type that needs a model
                     (a vision-capable one) to become searchable at all --
                     handled in index.py, not here. ingest.py's job stops
                     at "here are the raw image bytes and which page they're on."

Treating everything as one flat text blob (the naive approach) silently
loses the image entirely (raw PNG bytes aren't text) and often mangles
tables (reading a table row-by-row left-to-right as if it were a
sentence produces garbled, barely-searchable text). Splitting extraction
by element type is what makes each type retrievable in a way that
actually matches what it IS.
"""

import sys
from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF


@dataclass
class Element:
    kind: str            # "text" | "table" | "image"
    page: int             # 1-indexed page number, for citations
    content: str          # for "text"/"table": the extracted text.
                           # for "image": empty -- see image_bytes instead.
    source: str = ""       # the PDF filename this element came from --
                           # what makes a SHARED index across multiple
                           # PDFs possible: every element is tagged with
                           # which document it belongs to, so retrieval
                           # can search across all indexed PDFs at once,
                           # and citations can say which document an
                           # answer came from, not just which page.
    image_bytes: bytes | None = None   # only set when kind == "image"
    element_id: str = ""  # unique id, assigned in parse_pdf() -- namespaced
                           # by source (e.g. "sample.pdf::table-8") so
                           # elements from different PDFs never collide
                           # in a shared docstore/vector store.


def parse_pdf(pdf_path: Path) -> list[Element]:
    source = pdf_path.name
    doc = fitz.open(pdf_path)
    elements: list[Element] = []
    counter = 0

    for page_index in range(len(doc)):
        page = doc[page_index]
        page_num = page_index + 1

        # --- Tables first, so their text can be excluded from the plain
        # text block below (otherwise a table's cell text gets picked up
        # TWICE: once as a garbled table row and once inside a normal
        # text block covering the same page region).
        table_bboxes = []
        for table in page.find_tables().tables:
            rows = table.extract()
            if not rows or not any(any(cell for cell in row) for row in rows):
                continue
            # Render as a simple pipe-delimited grid -- readable as text,
            # keeps row/column structure legible to both a human and an
            # LLM, without needing markdown-table escaping.
            lines = [" | ".join(str(cell or "").strip() for cell in row) for row in rows]
            counter += 1
            elements.append(Element(
                kind="table",
                page=page_num,
                content="\n".join(lines),
                source=source,
                element_id=f"{source}::table-{counter}",
            ))
            table_bboxes.append(fitz.Rect(table.bbox))
            print(f"  [ingest] page {page_num}: found table ({len(rows)} rows)", file=sys.stderr)

        # --- Images
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            try:
                base_image = doc.extract_image(xref)
            except Exception:
                continue
            image_bytes = base_image["image"]
            # Skip tiny images (icons, bullet-point graphics, page
            # decorations) -- not worth a vision-model call, and they'd
            # just add noise to retrieval.
            if len(image_bytes) < 5000:
                continue
            counter += 1
            elements.append(Element(
                kind="image",
                page=page_num,
                content="",
                image_bytes=image_bytes,
                source=source,
                element_id=f"{source}::image-{counter}",
            ))
            print(f"  [ingest] page {page_num}: found image ({len(image_bytes)} bytes)", file=sys.stderr)

        # --- Plain text, with table regions excluded by actual bounding-box
        # overlap -- NOT a length/shape heuristic. An earlier version of
        # this function tried to filter out "table-looking" paragraphs by
        # length, which silently failed: pymupdf's plain "text" extraction
        # mode returns each table row as its OWN line, and several real
        # rows (e.g. "Northern Europe   58.2%   64.7%   +6.5 pp") are
        # longer than any reasonable length cutoff, so they passed the
        # filter and got duplicated as garbled text-kind elements right
        # alongside the properly-extracted table-kind element. The fix is
        # extracting text as BLOCKS (each with its own bbox) and dropping
        # any block whose bbox intersects a detected table's bbox --
        # geometry, not guesswork, is what correctly separates "this text
        # is part of the table" from "this text is a normal paragraph
        # that happens to be on the same page."
        blocks = page.get_text("blocks", sort=True)
        for block in blocks:
            x0, y0, x1, y1, block_text, *_ = block
            block_text = block_text.strip()
            if not block_text:
                continue
            block_rect = fitz.Rect(x0, y0, x1, y1)
            if any(block_rect.intersects(tb) for tb in table_bboxes):
                continue   # this block IS the table (or overlaps it) -- already captured as a table Element
            counter += 1
            elements.append(Element(
                kind="text",
                page=page_num,
                content=block_text,
                source=source,
                element_id=f"{source}::text-{counter}",
            ))

    doc.close()
    print(f"  [ingest] parsed {len(elements)} element(s) from {pdf_path.name}", file=sys.stderr)
    return elements
