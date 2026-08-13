"""
Generates sample_complex.pdf -- a harder stress-test document than
sample.pdf: more pages, THREE tables (not one), THREE charts (not one),
and multiple sections whose tables/charts are topically adjacent to each
other on purpose (e.g. two different tables both about "capacity," two
different charts both trending "over time") so retrieval and grading
have to discriminate WHICH table/chart actually answers a given
question, not just detect "a table exists somewhere." A single-table,
single-image document (sample.pdf) can't exercise that; this one can.

Run ONCE, same as generate_sample_pdf.py -- not part of the runtime
pipeline. Needs reportlab + matplotlib (see that file's docstring for
the isolated-venv setup).

Content: an expanded version of the same renewable-energy report,
covering three separate sub-domains (solar/wind capacity, storage
technology, workforce/investment) so each table and chart has a genuinely
different subject, even though they're all "renewable energy" broadly --
the same discrimination challenge stage05-rag-langgraph's Panchatantra
corpus poses with two different animal fables sharing a "carried by
another animal" plot shape.

Usage:
    pip install reportlab matplotlib   # or use .gen-venv, see generate_sample_pdf.py
    python3 generate_complex_pdf.py
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

OUTPUT_PATH = "sample_complex.pdf"


def _capacity_chart() -> io.BytesIO:
    years = [2019, 2020, 2021, 2022, 2023, 2024]
    solar = [4.2, 5.8, 8.1, 11.4, 15.9, 21.3]
    wind = [6.1, 7.4, 9.2, 11.8, 14.5, 17.9]
    fig, ax = plt.subplots(figsize=(6, 3.3))
    ax.plot(years, solar, marker="o", label="Solar", color="#e07b39")
    ax.plot(years, wind, marker="s", label="Wind", color="#3b7de0")
    ax.set_xlabel("Year")
    ax.set_ylabel("% of total grid capacity")
    ax.set_title("Solar vs. Wind Capacity, 2019-2024")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf


def _storage_chart() -> io.BytesIO:
    years = [2019, 2020, 2021, 2022, 2023, 2024]
    cost = [280, 240, 205, 165, 132, 98]
    fig, ax = plt.subplots(figsize=(6, 3.3))
    ax.bar(years, cost, color="#4c9a6d")
    ax.set_xlabel("Year")
    ax.set_ylabel("Battery storage cost ($/kWh)")
    ax.set_title("Grid-Scale Battery Storage Cost Decline, 2019-2024")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf


def _workforce_chart() -> io.BytesIO:
    categories = ["Manufacturing", "Installation", "R&D", "Grid Operations", "Policy/Admin"]
    jobs_2024 = [420, 680, 190, 310, 140]
    fig, ax = plt.subplots(figsize=(6, 3.3))
    ax.barh(categories, jobs_2024, color="#7b5ea7")
    ax.set_xlabel("Jobs added in 2024 (thousands)")
    ax.set_title("Renewable Sector Job Growth by Category, 2024")
    ax.grid(alpha=0.3, axis="x")
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
    h3 = ParagraphStyle("H3", parent=styles["Heading3"], spaceBefore=10, spaceAfter=6)
    body = ParagraphStyle("BodyX", parent=styles["BodyText"], fontSize=10.5, leading=15)
    caption = ParagraphStyle("Caption", parent=styles["BodyText"], fontSize=9, textColor=colors.grey)

    story = []

    # --- Page 1: title + overview -----------------------------------------
    story.append(Paragraph("Renewable Energy Sector: A Comprehensive 2024 Review", title_style))
    story.append(Spacer(1, 12))
    story.append(Paragraph(
        "This report covers three distinct dimensions of the renewable energy sector's 2024 "
        "performance: generation capacity growth (solar and wind), enabling storage technology "
        "cost trends, and workforce expansion across the sector's major job categories. Each "
        "dimension is analyzed separately, as growth in one does not necessarily track growth "
        "in the others.",
        body,
    ))
    story.append(Spacer(1, 10))
    story.append(Paragraph("1. Generation Capacity", h2))
    story.append(Paragraph("1.1 Overview", h3))
    story.append(Paragraph(
        "Renewable generation capacity has grown substantially faster than forecast in most "
        "major markets since 2022, driven by declining solar panel manufacturing costs and a "
        "wave of national subsidy programs following the 2022 global energy price shock. Wind "
        "capacity grew more steadily, constrained less by cost and more by offshore permitting "
        "timelines.",
        body,
    ))
    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "The chart below shows the divergence between solar and wind capacity growth over the "
        "six-year study window. Solar, which trailed wind for most of the period, overtook it "
        "in 2024 for the first time.",
        body,
    ))
    story.append(Spacer(1, 10))
    story.append(Image(_capacity_chart(), width=5.5 * inch, height=3.0 * inch))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "<i>Figure 1: Global installed solar and wind capacity as a percentage of total grid "
        "capacity, 2019-2024.</i>", caption,
    ))

    story.append(PageBreak())

    # --- Page 2: regional table ---------------------------------------------
    story.append(Paragraph("1.2 Regional Adoption Rates", h3))
    story.append(Paragraph(
        "Adoption rates vary significantly by region. The table below reports each region's "
        "renewable share of total electricity generation for 2023 and 2024.",
        body,
    ))
    story.append(Spacer(1, 10))
    regional_data = [
        ["Region", "2023 Share", "2024 Share", "YoY Change"],
        ["Northern Europe", "58.2%", "64.7%", "+6.5 pp"],
        ["East Asia", "31.4%", "39.1%", "+7.7 pp"],
        ["North America", "27.8%", "32.0%", "+4.2 pp"],
        ["South Asia", "19.6%", "24.3%", "+4.7 pp"],
        ["Sub-Saharan Africa", "14.1%", "16.8%", "+2.7 pp"],
        ["Oceania", "35.0%", "41.5%", "+6.5 pp"],
    ]
    tbl1 = Table(regional_data, colWidths=[1.8 * inch, 1.3 * inch, 1.3 * inch, 1.1 * inch])
    tbl1.setStyle(TableStyle([
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
    story.append(tbl1)
    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "East Asia recorded the largest year-over-year gain at 7.7 percentage points, driven "
        "primarily by accelerated solar deployment.",
        body,
    ))

    story.append(PageBreak())

    # --- Page 3: storage technology ------------------------------------------
    story.append(Paragraph("2. Storage Technology", h2))
    story.append(Paragraph(
        "Grid-scale battery storage cost has fallen sharply over the study period, a critical "
        "enabler for renewable adoption since storage capacity determines how much variable "
        "solar and wind generation a grid can absorb without curtailment.",
        body,
    ))
    story.append(Spacer(1, 10))
    story.append(Image(_storage_chart(), width=5.5 * inch, height=3.0 * inch))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "<i>Figure 2: Grid-scale lithium-ion battery storage cost per kWh, 2019-2024.</i>", caption,
    ))
    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "Storage cost declined from $280/kWh in 2019 to $98/kWh in 2024, a 65% reduction. "
        "Analysts attribute roughly half of this decline to manufacturing scale and half to "
        "chemistry improvements in cathode materials.",
        body,
    ))
    story.append(Spacer(1, 10))
    story.append(Paragraph("2.1 Storage Deployment by Technology Type", h3))
    story.append(Paragraph(
        "The table below breaks down 2024 grid-scale storage deployment by technology, "
        "measured in gigawatt-hours (GWh) of installed capacity.",
        body,
    ))
    story.append(Spacer(1, 10))
    storage_data = [
        ["Technology", "2023 GWh Installed", "2024 GWh Installed", "Growth"],
        ["Lithium-ion", "142", "218", "+53.5%"],
        ["Flow battery", "8", "19", "+137.5%"],
        ["Pumped hydro", "310", "324", "+4.5%"],
        ["Compressed air", "3", "7", "+133.3%"],
    ]
    tbl2 = Table(storage_data, colWidths=[1.8 * inch, 1.6 * inch, 1.6 * inch, 1.0 * inch])
    tbl2.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2b5540")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0f2f5")]),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(tbl2)
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "Flow battery and compressed-air storage grew fastest in percentage terms, though both "
        "remain small in absolute capacity compared to pumped hydro and lithium-ion.",
        body,
    ))

    story.append(PageBreak())

    # --- Page 4: workforce ----------------------------------------------------
    story.append(Paragraph("3. Workforce and Investment", h2))
    story.append(Paragraph(
        "Sector employment grew across all major job categories in 2024, with installation "
        "roles adding the most positions in absolute terms.",
        body,
    ))
    story.append(Spacer(1, 10))
    story.append(Image(_workforce_chart(), width=5.5 * inch, height=3.0 * inch))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "<i>Figure 3: New renewable-sector jobs added by category, 2024, in thousands.</i>", caption,
    ))
    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "Installation added 680,000 jobs in 2024, more than any other category, reflecting the "
        "labor-intensive nature of solar and wind construction relative to manufacturing, which "
        "benefits more from automation.",
        body,
    ))
    story.append(Spacer(1, 10))
    story.append(Paragraph("3.1 Regional Investment Totals", h3))
    story.append(Paragraph(
        "Total capital investment by region in 2024, in billions of US dollars.",
        body,
    ))
    story.append(Spacer(1, 10))
    investment_data = [
        ["Region", "2023 Investment ($B)", "2024 Investment ($B)", "Growth"],
        ["Northern Europe", "84", "112", "+33.3%"],
        ["East Asia", "156", "231", "+48.1%"],
        ["North America", "98", "127", "+29.6%"],
        ["South Asia", "41", "58", "+41.5%"],
    ]
    tbl3 = Table(investment_data, colWidths=[1.8 * inch, 1.6 * inch, 1.6 * inch, 1.0 * inch])
    tbl3.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#5a3a2b")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0f2f5")]),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(tbl3)
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "East Asia recorded both the highest absolute investment and the highest percentage "
        "growth in 2024, consistent with its lead in regional capacity adoption growth "
        "described in Section 1.2.",
        body,
    ))

    story.append(PageBreak())

    # --- Page 5: outlook --------------------------------------------------------
    story.append(Paragraph("4. Outlook", h2))
    story.append(Paragraph(
        "Analysts project continued acceleration through 2026 across all three dimensions "
        "covered in this report. Solar capacity growth is expected to keep outpacing wind due "
        "to shorter installation timelines. Storage costs are projected to fall below $80/kWh "
        "by 2026 as flow battery manufacturing reaches commercial scale. Workforce growth is "
        "expected to shift increasingly toward grid operations roles as installed base "
        "maintenance needs grow relative to new construction.",
        body,
    ))

    doc.build(story)
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    build_pdf()
