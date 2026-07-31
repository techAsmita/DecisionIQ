"""
modules/reports.py
====================
Executive Report Generator: packages KPIs, decision engine findings,
the forecast, and (optionally) an AI narrative into a downloadable PDF.

Design decisions:
- This module REUSES analytics.compute_kpis(), decision_engine.run_decision_engine(),
  and forecasting.train_forecast_model() rather than recomputing anything —
  the PDF and the live dashboard are guaranteed to show the same numbers
  because they call the exact same functions.
- Charts are rendered via matplotlib (not Plotly) purely for the PDF,
  since ReportLab embeds static images, not interactive JS charts.
  Plotly stays the tool for the live Streamlit dashboard; matplotlib is
  only used for this static export — each tool used for what it's best at.
- Returns raw PDF bytes rather than writing to a fixed path, so
  st.download_button in app.py can offer it directly without a temp file
  the user has to manage.
"""

import io
from datetime import datetime

import matplotlib
matplotlib.use("Agg")  # non-interactive backend, required for server-side rendering
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
)

from modules import analytics
from modules import decision_engine
from modules import forecasting


def _revenue_trend_image(df) -> io.BytesIO:
    """Render the revenue trend as a static matplotlib PNG for PDF embedding."""
    d = df.copy()
    import pandas as pd
    d["Date"] = pd.to_datetime(d["Date"])
    monthly = d.set_index("Date").resample("ME")["Revenue"].sum()

    fig, ax = plt.subplots(figsize=(6.5, 3))
    ax.plot(monthly.index, monthly.values, marker="o", color="#2563EB")
    ax.set_title("Revenue Trend (Monthly)")
    ax.set_ylabel("Revenue ($)")
    fig.autofmt_xdate()
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf


def generate_executive_pdf(df, ai_summary: str = None) -> bytes:
    """
    Build the full executive PDF report and return it as bytes.
    ai_summary is optional — the report is still complete without it,
    since AI narrative generation requires a configured Gemini key.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter,
                             topMargin=0.6 * inch, bottomMargin=0.6 * inch)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("TitleStyle", parent=styles["Title"], fontSize=22)
    h2_style = ParagraphStyle("H2Style", parent=styles["Heading2"], spaceBefore=14)
    body_style = styles["BodyText"]

    story = []

    # --- Cover / header ---
    story.append(Paragraph("DecisionIQ Executive Report", title_style))
    story.append(Paragraph("AI-Powered Decision Intelligence for Pharmaceutical Sales", body_style))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y')}", body_style))
    story.append(Spacer(1, 0.3 * inch))

    # --- KPIs ---
    kpis = analytics.compute_kpis(df)
    story.append(Paragraph("Key Performance Indicators", h2_style))

    kpi_rows = [
        ["Total Revenue", f"${kpis['total_revenue']:,.0f}"],
        ["Total Profit", f"${kpis['total_profit']:,.0f}"],
        ["Profit Margin", f"{kpis['profit_margin']:.1f}%"],
        ["Total Orders", f"{kpis['total_orders']:,}"],
        ["Avg Order Value", f"${kpis['avg_order_value']:,.0f}"],
        ["Best Region", kpis["best_region"]],
        ["Worst Region", kpis["worst_region"]],
        ["Top Therapy Area", kpis["top_therapy"]],
        ["Top Drug", kpis["top_drug"]],
        ["Monthly Growth", f"{kpis['monthly_growth']:.1f}%"],
        ["Quarterly Growth", f"{kpis['quarterly_growth']:.1f}%"],
    ]
    kpi_table = Table(kpi_rows, colWidths=[2.5 * inch, 3 * inch])
    kpi_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F1F5F9")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(kpi_table)
    story.append(Spacer(1, 0.25 * inch))

    # --- Revenue trend chart ---
    story.append(Paragraph("Revenue Trend", h2_style))
    chart_buf = _revenue_trend_image(df)
    story.append(Image(chart_buf, width=6.5 * inch, height=3 * inch))

    # --- Forecast ---
    story.append(Paragraph("Predictive Forecast", h2_style))
    forecast_result = forecasting.train_forecast_model(df)
    if forecast_result["success"]:
        story.append(Paragraph(
            f"Next month's revenue is forecast at "
            f"<b>${forecast_result['next_month_forecast']:,.0f}</b> "
            f"(model RMSE: ${forecast_result['rmse']:,.0f}, "
            f"R²: {forecast_result['r2']:.3f}).", body_style))
    else:
        story.append(Paragraph(forecast_result["reason"], body_style))

    # --- Decision Engine findings ---
    story.append(PageBreak())
    story.append(Paragraph("Business Risks & Recommendations", h2_style))
    issues = decision_engine.run_decision_engine(df)

    if not issues:
        story.append(Paragraph("No significant issues detected in the current dataset.", body_style))
    else:
        for issue in issues:
            story.append(Paragraph(
                f"<b>[{issue['severity']}] {issue['category']}</b>", body_style))
            story.append(Paragraph(f"Finding: {issue['finding']}", body_style))
            story.append(Paragraph(f"Recommendation: {issue['recommendation']}", body_style))
            story.append(Spacer(1, 0.12 * inch))

    # --- AI narrative (optional) ---
    if ai_summary:
        story.append(Paragraph("AI Business Consultant Summary", h2_style))
        for line in ai_summary.split("\n"):
            if line.strip():
                story.append(Paragraph(line.strip(), body_style))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()
