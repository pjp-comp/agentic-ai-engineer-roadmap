"""
Generates sample.pdf -- the realistic, multi-section source document this
example's RAG pipeline indexes and queries. Run ONCE to produce the PDF;
not part of the example's runtime pipeline (build_index.py / agent.py
never import this file). Needs reportlab + matplotlib, installed
separately from the example's own pyproject.toml dependencies since
they're a one-time authoring tool, not something agent.py needs at
query time.

The PDF deliberately has all three element types real RAG systems have
to handle differently (this repo's stage05-multimodal-rag-agent README
explains why they need different treatment):
  - Real paragraph text (multiple pages, multiple sections)
  - A real DATA TABLE (drawn as an actual PDF table via reportlab, with
    real rows/columns/borders -- extractable as structured text)
  - A real embedded CHART IMAGE (a matplotlib bar chart, saved as PNG,
    embedded into the PDF as an actual image XObject -- extractable only
    via a vision model, not text extraction)

Content: a fictional research report on renewable energy adoption,
chosen because it naturally wants a table (adoption rates by country) and
a chart (a trend over time) alongside real prose -- not shoehorned in.

Usage:
    pip install reportlab matplotlib
    python3 generate_sample_pdf.py
"""

import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak,
)

OUTPUT_PATH = "sample.pdf"


def _build_chart_image() -> io.BytesIO:
    """A real chart -- this is the part of the PDF a text extractor
    cannot read at all. Only a vision model (or a human) can describe
    what this actually shows.
    """
    years = [2019, 2020, 2021, 2022, 2023, 2024]
    solar = [4.2, 5.8, 8.1, 11.4, 15.9, 21.3]
    wind = [6.1, 7.4, 9.2, 11.8, 14.5, 17.9]

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(years, solar, marker="o", label="Solar", color="#e07b39")
    ax.plot(years, wind, marker="s", label="Wind", color="#3b7de0")
    ax.set_xlabel("Year")
    ax.set_ylabel("Global installed capacity (% of total grid)")
    ax.set_title("Solar vs. Wind Adoption, 2019-2024")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf


def build_pdf() -> None:
    doc = SimpleDocTemplate(OUTPUT_PATH, pagesize=letter,
                             topMargin=0.75 * inch, bottomMargin=0.75 * inch)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("TitleX", parent=styles["Title"], fontSize=18)
    h2 = ParagraphStyle("H2", parent=styles["Heading2"], spaceBefore=14, spaceAfter=8)
    body = ParagraphStyle("BodyX", parent=styles["BodyText"], fontSize=10.5, leading=15)

    story = []

    # --- Page 1: title + intro text --------------------------------------
    story.append(Paragraph("Renewable Energy Adoption: A 2024 Sector Review", title_style))
    story.append(Spacer(1, 12))
    story.append(Paragraph(
        "This report summarizes global renewable energy adoption trends between 2019 and 2024, "
        "with a focus on solar and wind capacity growth, regional deployment differences, and "
        "policy drivers behind the acceleration observed in the final two years of the study period.",
        body,
    ))
    story.append(Spacer(1, 10))
    story.append(Paragraph("1. Overview", h2))
    story.append(Paragraph(
        "Renewable energy capacity has grown substantially faster than forecast in most major "
        "markets since 2022. Two factors are most frequently cited by analysts: a sharp decline "
        "in solar panel manufacturing costs driven by expanded production in East Asia, and a wave "
        "of national subsidy programs introduced following the 2022 global energy price shock. Wind "
        "capacity grew more steadily across the same period, constrained less by cost and more by "
        "permitting timelines for offshore installations.",
        body,
    ))
    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "The chart below illustrates the divergence between solar and wind growth rates over the "
        "six-year study window. Solar capacity, which trailed wind for most of the period, "
        "overtook it in 2024 for the first time.",
        body,
    ))
    story.append(Spacer(1, 12))

    chart_buf = _build_chart_image()
    story.append(Image(chart_buf, width=5.5 * inch, height=3.2 * inch))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "<i>Figure 1: Global installed solar and wind capacity as a percentage of total grid "
        "capacity, 2019-2024. Solar overtakes wind for the first time in 2024.</i>",
        ParagraphStyle("Caption", parent=styles["BodyText"], fontSize=9, textColor=colors.grey),
    ))

    story.append(PageBreak())

    # --- Page 2: regional table --------------------------------------------
    story.append(Paragraph("2. Regional Adoption Rates", h2))
    story.append(Paragraph(
        "Adoption rates vary significantly by region, driven largely by differences in national "
        "subsidy structures and grid infrastructure readiness. The table below reports each "
        "region's renewable share of total electricity generation for 2023 and 2024, along with "
        "the year-over-year percentage point change.",
        body,
    ))
    story.append(Spacer(1, 12))

    table_data = [
        ["Region", "2023 Renewable Share", "2024 Renewable Share", "YoY Change"],
        ["Northern Europe", "58.2%", "64.7%", "+6.5 pp"],
        ["East Asia", "31.4%", "39.1%", "+7.7 pp"],
        ["North America", "27.8%", "32.0%", "+4.2 pp"],
        ["South Asia", "19.6%", "24.3%", "+4.7 pp"],
        ["Sub-Saharan Africa", "14.1%", "16.8%", "+2.7 pp"],
        ["Oceania", "35.0%", "41.5%", "+6.5 pp"],
    ]
    tbl = Table(table_data, colWidths=[1.8 * inch, 1.5 * inch, 1.5 * inch, 1.1 * inch])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2b3a55")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0f2f5")]),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 12))
    story.append(Paragraph(
        "East Asia recorded the largest year-over-year gain at 7.7 percentage points, attributed "
        "primarily to accelerated solar deployment. Sub-Saharan Africa recorded the smallest gain, "
        "constrained by limited grid infrastructure investment relative to installed generation capacity.",
        body,
    ))

    story.append(PageBreak())

    # --- Page 3: policy discussion -----------------------------------------
    story.append(Paragraph("3. Policy Drivers", h2))
    story.append(Paragraph(
        "Three policy mechanisms appear most strongly correlated with above-average regional "
        "adoption growth in this study: direct capital subsidies for residential and utility-scale "
        "solar installation, streamlined permitting for offshore wind projects, and carbon pricing "
        "schemes that raise the relative cost of fossil generation.",
        body,
    ))
    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "Northern Europe's leading position is attributed largely to a combination of all three "
        "mechanisms operating simultaneously since 2021, alongside grid infrastructure that was "
        "already well-suited to variable renewable input before the acceleration began. East Asia's "
        "rapid 2024 gains are attributed primarily to manufacturing-driven cost declines rather than "
        "policy incentives, since several of the region's largest markets introduced no new subsidy "
        "programs during the study period.",
        body,
    ))
    story.append(Spacer(1, 10))
    story.append(Paragraph("4. Outlook", h2))
    story.append(Paragraph(
        "Analysts project continued acceleration through 2026, with solar capacity growth expected "
        "to outpace wind for the remainder of the decade due to shorter installation timelines and "
        "continued cost declines. Offshore wind is expected to see renewed growth beginning in 2026 "
        "as a wave of projects currently in permitting reach construction.",
        body,
    ))

    doc.build(story)
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    build_pdf()
