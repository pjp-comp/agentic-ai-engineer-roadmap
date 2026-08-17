"""
CHUNKING -- element-aware parsing for every file type this pipeline
accepts. Turns a source file into a list of typed Elements (text blocks,
tables, images), each tagged with the page (or 1, for page-less formats)
it came from. This is the FIRST stage of the pipeline (see vision.py for
the second: generating a searchable summary per element).

Four source types, four different extraction paths -- see parse_file()
for the dispatcher that picks one by file extension:

  .pdf              -> parse_pdf(): the full element-aware extraction
                        this module started with (see below).
  .txt / .md        -> parse_text_file(): the file's content, read
                        directly. No parsing library, no model call --
                        it's already exactly the text it is.
  .png/.jpg/.jpeg/
  .webp             -> parse_image_file(): the ENTIRE file is one image
                        Element, no PDF page or embedding context around
                        it to draw a caption from (see caption handling
                        below) -- it goes straight to vision.py's
                        vision-model summarization, same as an image
                        found INSIDE a PDF would.

Why element-aware for PDFs specifically, not "extract all text into one
blob": a PDF's table and a PDF's embedded chart are fundamentally
different kinds of content that need different extraction paths:

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
                     handled in vision.py, not here. chunking.py's job
                     stops at "here are the raw image bytes and which
                     page they're on."

Treating everything as one flat text blob (the naive approach) silently
loses images entirely (raw PNG bytes aren't text) and often mangles
tables (reading a table row-by-row left-to-right as if it were a
sentence produces garbled, barely-searchable text). Splitting extraction
by element type is what makes each type retrievable in a way that
actually matches what it IS.
"""

import sys
from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF

CAPTION_WINDOW_PT = 80   # ~1 paragraph of vertical space, in PDF points
MIN_IMAGE_BYTES = 5000   # below this, treat as a decoration/icon, not content

TEXT_EXTENSIONS = {".txt", ".md"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


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
    caption: str = ""     # only set when kind == "image" -- nearby text on
                           # the same page, passed to the vision model as
                           # GROUNDING context (see vision.py's
                           # summarize_image). A vision model describing a
                           # chart from pixels alone can hallucinate
                           # specifics (wrong numbers, invented categories,
                           # invented trends) -- the surrounding prose
                           # usually states the real numbers in words, so
                           # giving the model that text alongside the image
                           # measurably reduces fabrication.
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
            print(f"  [chunking] page {page_num}: found table ({len(rows)} rows)", file=sys.stderr)

        # --- Images. Each image gets a NEARBY-text caption for grounding
        # the vision model in vision.py -- not the whole page's text. A
        # real bug this caught: a page with both a chart AND an unrelated
        # table below it can let the vision model latch onto the OTHER
        # table's numbers and misattribute them as chart categories, if
        # given the whole page. Restricting to text within a fixed
        # vertical window of the image's own bounding box (roughly one
        # paragraph above and below) keeps the grounding text limited to
        # what a human reader would actually associate with THIS image,
        # not everything else that happens to share the page.
        all_blocks = page.get_text("blocks", sort=True)

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
            if len(image_bytes) < MIN_IMAGE_BYTES:
                continue

            try:
                image_rects = page.get_image_rects(xref)
                image_bbox = image_rects[0] if image_rects else None
            except Exception:
                image_bbox = None

            if image_bbox is not None:
                nearby_lines = [
                    block_text.strip()
                    for x0, y0, x1, y1, block_text, *_ in all_blocks
                    if block_text.strip()
                    and y0 >= image_bbox.y0 - CAPTION_WINDOW_PT
                    and y1 <= image_bbox.y1 + CAPTION_WINDOW_PT
                ]
                caption_text = "\n".join(nearby_lines)
            else:
                caption_text = ""   # couldn't locate the image on the page -- ground with nothing rather than the whole page

            counter += 1
            elements.append(Element(
                kind="image",
                page=page_num,
                content="",
                image_bytes=image_bytes,
                caption=caption_text,
                source=source,
                element_id=f"{source}::image-{counter}",
            ))
            print(f"  [chunking] page {page_num}: found image ({len(image_bytes)} bytes)", file=sys.stderr)

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
    print(f"  [chunking] parsed {len(elements)} element(s) from {pdf_path.name}", file=sys.stderr)
    return elements


def parse_text_file(path: Path) -> list[Element]:
    """A .txt/.md file needs no library and no model call to become an
    Element -- it's already exactly the text it is. Split on blank lines
    into paragraph-sized chunks, the same granularity a PDF's per-block
    text extraction produces, rather than indexing the whole file as one
    giant Element (which would embed poorly for the same reason a whole
    PDF page embedded as one blob would -- see this module's docstring).
    """
    source = path.name
    raw = path.read_text(encoding="utf-8", errors="replace")
    elements: list[Element] = []
    counter = 0

    for para in raw.split("\n\n"):
        para = para.strip()
        if not para:
            continue
        counter += 1
        elements.append(Element(
            kind="text",
            page=1,   # text files have no page concept -- always 1, page is a PDF-specific citation detail
            content=para,
            source=source,
            element_id=f"{source}::text-{counter}",
        ))

    print(f"  [chunking] parsed {len(elements)} element(s) from {path.name}", file=sys.stderr)
    return elements


def parse_image_file(path: Path) -> list[Element]:
    """A standalone image file (not embedded in a PDF) becomes ONE image
    Element, with no caption -- there's no surrounding page text to draw
    a caption window from the way chunking.py's PDF path does (see
    CAPTION_WINDOW_PT above). vision.py's summarize_image() still works
    correctly with an empty caption; it just loses the extra grounding
    that nearby prose would otherwise provide (see vision.py's own
    docstring for why grounding helps when it's available).
    """
    source = path.name
    image_bytes = path.read_bytes()
    element = Element(
        kind="image",
        page=1,
        content="",
        image_bytes=image_bytes,
        caption="",   # no page/document context to draw a caption from -- see docstring above
        source=source,
        element_id=f"{source}::image-1",
    )
    print(f"  [chunking] parsed 1 element(s) from {path.name} ({len(image_bytes)} bytes)", file=sys.stderr)
    return [element]


def parse_file(path: Path) -> list[Element]:
    """The dispatcher build_index.py calls -- picks the right parser by
    file extension so callers don't need to know or care which one a
    given file needs. Raises ValueError for anything unrecognized rather
    than silently skipping it, so an unsupported file in assets/ is a
    loud, immediate error at index time, not a silent gap discovered
    later when a question about it can't be answered.
    """
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return parse_pdf(path)
    if suffix in TEXT_EXTENSIONS:
        return parse_text_file(path)
    if suffix in IMAGE_EXTENSIONS:
        return parse_image_file(path)
    raise ValueError(
        f"Unsupported file type {suffix!r} for {path.name}. "
        f"Supported: .pdf, {', '.join(sorted(TEXT_EXTENSIONS))}, {', '.join(sorted(IMAGE_EXTENSIONS))}"
    )
