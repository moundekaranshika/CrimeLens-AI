"""PDF intelligence report generator for CrimeLens AI."""

from __future__ import annotations

import io
from datetime import datetime
from typing import Any, Optional

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

NAVY = colors.HexColor("#1a237e")
LIGHT_BLUE = colors.HexColor("#4fc3f7")
RED_ALERT = colors.HexColor("#e53935")


def _build_styles() -> dict:
    """Create custom paragraph styles for the report."""
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "Title",
            parent=base["Title"],
            fontSize=22,
            textColor=NAVY,
            alignment=TA_CENTER,
            spaceAfter=12,
        ),
        "subtitle": ParagraphStyle(
            "Subtitle",
            parent=base["Normal"],
            fontSize=12,
            textColor=colors.grey,
            alignment=TA_CENTER,
            spaceAfter=20,
        ),
        "heading": ParagraphStyle(
            "Heading",
            parent=base["Heading2"],
            fontSize=14,
            textColor=NAVY,
            spaceBefore=16,
            spaceAfter=8,
        ),
        "body": ParagraphStyle(
            "Body",
            parent=base["Normal"],
            fontSize=10,
            leading=14,
            alignment=TA_LEFT,
        ),
        "bullet": ParagraphStyle(
            "Bullet",
            parent=base["Normal"],
            fontSize=10,
            leading=14,
            leftIndent=20,
            bulletIndent=10,
        ),
    }


def generate_intelligence_report(
    df: pd.DataFrame,
    kpis: dict[str, Any],
    network_insights: list[str],
    risk_forecast: pd.DataFrame,
    anomalies: pd.DataFrame,
    ai_recommendations: Optional[list[str]] = None,
) -> bytes:
    """
    Generate a downloadable PDF intelligence report.

    Returns PDF content as bytes.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=50,
        leftMargin=50,
        topMargin=50,
        bottomMargin=50,
    )
    styles = _build_styles()
    story: list = []

    # Header
    story.append(Paragraph("CrimeLens AI", styles["title"]))
    story.append(Paragraph("Karnataka State Police — Intelligence Report", styles["subtitle"]))
    story.append(Paragraph(
        f"Generated: {datetime.now().strftime('%d %B %Y, %H:%M IST')}",
        styles["subtitle"],
    ))
    story.append(HRFlowable(width="100%", thickness=2, color=NAVY))
    story.append(Spacer(1, 0.2 * inch))

    # Crime Summary
    story.append(Paragraph("1. Crime Summary", styles["heading"]))
    summary_data = [
        ["Metric", "Value"],
        ["Total Crimes", str(kpis.get("total_crimes", 0))],
        ["Active Cases", str(kpis.get("active_cases", 0))],
        ["Arrests Made", str(kpis.get("arrests_made", 0))],
        ["Arrest Rate", f"{kpis.get('arrest_rate', 0)}%"],
        ["Repeat Offenders", str(kpis.get("repeat_offenders", 0))],
        ["High Risk District", f"{kpis.get('high_risk_district', 'N/A')} ({kpis.get('high_risk_count', 0)})"],
    ]
    summary_table = Table(summary_data, colWidths=[3 * inch, 3 * inch])
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f5f5")]),
        ("PADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 0.15 * inch))

    # Hotspots
    story.append(Paragraph("2. Crime Hotspots", styles["heading"]))
    if len(df):
        hotspots = df.groupby(["district", "crime_type"]).size().reset_index(name="count")
        hotspots = hotspots.sort_values("count", ascending=False).head(10)
        hotspot_data = [["District", "Crime Type", "Count"]] + hotspots.values.tolist()
        hotspot_table = Table(hotspot_data, colWidths=[2 * inch, 2 * inch, 1.5 * inch])
        hotspot_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), LIGHT_BLUE),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("PADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(hotspot_table)
    else:
        story.append(Paragraph("No hotspot data available.", styles["body"]))
    story.append(Spacer(1, 0.15 * inch))

    # Network Findings
    story.append(Paragraph("3. Network Analysis Findings", styles["heading"]))
    for insight in network_insights[:8]:
        story.append(Paragraph(f"• {insight}", styles["bullet"]))
    story.append(Spacer(1, 0.15 * inch))

    # Predicted Risks
    story.append(Paragraph("4. Predicted District Risk Rankings", styles["heading"]))
    if len(risk_forecast):
        risk_data = [["Rank", "District", "Risk Score", "Level", "Trend"]]
        for _, row in risk_forecast.head(10).iterrows():
            risk_data.append([
                str(int(row["risk_rank"])),
                str(row["district"]),
                f"{row['forecast_risk_score']:.3f}",
                str(row["risk_level"]),
                str(row["trend"]),
            ])
        risk_table = Table(risk_data, colWidths=[0.6 * inch, 2 * inch, 1 * inch, 1.2 * inch, 1 * inch])
        risk_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), NAVY),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("PADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(risk_table)
    story.append(Spacer(1, 0.15 * inch))

    # Anomalies
    story.append(Paragraph("5. Anomaly Alerts", styles["heading"]))
    if len(anomalies) and "is_anomaly" in anomalies.columns:
        anomaly_rows = anomalies[anomalies["is_anomaly"]].head(5)
        if len(anomaly_rows):
            for _, row in anomaly_rows.iterrows():
                story.append(Paragraph(
                    f"• {row['district']} ({row['month_year']}): "
                    f"{int(row['crime_count'])} crimes — {row.get('severity', 'Alert')}",
                    styles["bullet"],
                ))
        else:
            story.append(Paragraph("No critical anomalies detected in current period.", styles["body"]))
    else:
        story.append(Paragraph("Anomaly data unavailable.", styles["body"]))
    story.append(Spacer(1, 0.15 * inch))

    # Strategic Recommendations
    story.append(Paragraph("6. Strategic Recommendations", styles["heading"]))
    recommendations = ai_recommendations or [
        "Increase patrol density in top 3 high-risk districts during peak hours.",
        "Deploy cybercrime task force to districts with rising fraud incidents.",
        "Prioritize repeat offender surveillance using network centrality scores.",
        "Coordinate inter-district operations for identified organized crime clusters.",
        "Review anomaly alerts weekly for emerging crime trend response.",
    ]
    for rec in recommendations:
        story.append(Paragraph(f"• {rec}", styles["bullet"]))

    story.append(Spacer(1, 0.3 * inch))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.grey))
    story.append(Paragraph(
        "<i>CONFIDENTIAL — For Official Use Only | Karnataka State Police | CrimeLens AI v1.0</i>",
        ParagraphStyle("Footer", parent=styles["body"], fontSize=8, textColor=colors.grey, alignment=TA_CENTER),
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()
