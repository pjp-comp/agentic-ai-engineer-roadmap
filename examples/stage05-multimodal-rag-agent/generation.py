"""
GENERATION -- builds the final answer from graded-relevant content.
Isolated from retrieval.py so "did we find the right thing" and "did we
answer it correctly" stay separately testable.

RAW content here, not summaries -- the entire point of the multi-vector
split (see vision.py): the summary is a LOSSY compression optimized for
matching a QUESTION's phrasing, not for actually answering it -- e.g. a
table summary says "shows adoption rates by region" but the actual
NUMBERS live only in the raw table text. Retrieval and generation
deliberately read different representations of the same element for
this reason.
"""

import ollama

from vision import LOCAL_MODEL


def compute_table_totals(table_text: str) -> str:
    """Parses a table's raw pipe-delimited text (the format chunking.py's
    parse_pdf() produces -- "Header | Header | ...\\nRow | Row | ...")
    back into columns, and computes a real, Python-arithmetic sum for
    every column that's actually numeric. Returns a short block of
    precomputed facts, or "" if the table has no summable numeric column.

    Why this exists: llama3.1:8b (like most small local LLMs) is
    unreliable at multi-step addition done as free-text reasoning inside
    a single generation call -- a real, verified failure: asked to total
    a 4-row investment column (112+231+127+58, which is 528), it visibly
    fumbled the arithmetic inline and produced 428, then contradicted
    itself trying to justify that number in the same response. The raw
    table CONTEXT it was given was completely correct -- this wasn't a
    retrieval or grading bug, it was the model doing arithmetic in its
    head and getting it wrong.

    The fix is NOT "ask the model to be more careful" (prompting can't
    fix an LLM's unreliable mental arithmetic) and not a general
    tool-calling loop (a bigger change than this one failure mode needs).
    It's simpler: compute the numbers Python actually knows how to
    compute correctly, BEFORE the LLM call, and hand them over as
    pre-computed facts the model only has to STATE, never calculate.
    generate() below appends this block to context for every table in
    the graded-relevant set.
    """
    lines = [line for line in table_text.strip().split("\n") if line.strip()]
    if len(lines) < 2:
        return ""

    headers = [h.strip() for h in lines[0].split("|")]
    rows = [[cell.strip() for cell in line.split("|")] for line in lines[1:]]

    facts = []
    for col_index, header in enumerate(headers):
        # Rate/ratio columns aren't summable -- adding "growth %" or
        # "share %" across regions produces a number that LOOKS like a
        # total but means nothing (percentages of different bases don't
        # add up to a meaningful combined percentage). Every table in
        # this example's corpus has exactly this kind of column
        # ("YoY Change", "Growth", "2024 Renewable Share"), so this
        # exclusion is load-bearing, not defensive over-engineering.
        header_lower = header.lower()
        if "%" in header_lower or "growth" in header_lower or "change" in header_lower or "share" in header_lower:
            continue

        values = []
        for row in rows:
            if col_index >= len(row):
                continue
            # Strip common formatting so "231", "$231", "231%", "+231 pp"
            # all parse as numbers -- a real table mixes plain numbers,
            # currency, and percentages in different columns.
            cleaned = row[col_index].replace("$", "").replace("%", "").replace("pp", "").replace(",", "").replace("+", "").strip()
            try:
                values.append(float(cleaned))
            except ValueError:
                values.append(None)

        numeric_values = [v for v in values if v is not None]
        # Only compute a sum if EVERY row in this column parsed as a
        # number -- a column that's mostly numbers with one text cell
        # (e.g. a "Notes" column with one stray number) isn't a real
        # numeric column, and summing it would produce a meaningless total.
        if numeric_values and len(numeric_values) == len(rows):
            total = sum(numeric_values)
            # Keep it as an int when the total has no fractional part --
            # "528" reads as a real total; "528.0" reads like a rounding artifact.
            total_str = str(int(total)) if total == int(total) else f"{total:.2f}"
            facts.append(f"Sum of \"{header}\" column across all {len(rows)} rows: {total_str}")

    if not facts:
        return ""
    return "Precomputed totals for this table (calculated in code, not by you -- trust these exactly):\n" + "\n".join(facts)


def generate(graded_relevant: list[dict], original_query: str) -> str:
    # Each chunk is tagged with its SOURCE document, not just its page --
    # necessary once the shared index holds more than one PDF, so an
    # answer can say which document it came from, not just where in "the"
    # document (there may be several).
    context_blocks = []
    for c in graded_relevant:
        block = f"[{c['source']}, {c['kind']}, page {c['page']}]\n{c['raw']}"
        if c["kind"] == "table":
            totals = compute_table_totals(c["raw"])
            if totals:
                block += f"\n\n{totals}"
        context_blocks.append(block)
    context = "\n\n".join(context_blocks)

    response = ollama.chat(
        model=LOCAL_MODEL,
        messages=[{
            "role": "user",
            "content": (
                f"Context from one or more reports (tables and image descriptions included):\n{context}\n\n"
                f"Question: {original_query}\n\n"
                "Answer using ONLY the context above, in 1-3 sentences. Do not repeat the raw "
                "table or list every row -- state the specific answer directly. If a question "
                "asks for a total/sum and a \"Precomputed totals\" block is present, use THAT "
                "number exactly -- do not recalculate it yourself, and do not show your work. "
                "Cite the source document and page, e.g. (sample.pdf, page 2). If the context "
                "above doesn't actually contain the answer, say plainly that the document(s) "
                "don't cover this -- do not comment on the context itself or ask for more information. "
                "Do not invent a number that isn't stated in the context or a Precomputed totals block."
            ),
        }],
        think=False,
        options={"temperature": 0},
    )
    return response.message.content


def refuse() -> str:
    return (
        "I don't have enough information in the document to answer this question "
        "confidently, even after rewriting the query."
    )
