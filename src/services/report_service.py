from __future__ import annotations

from io import BytesIO
from typing import Any

import pandas as pd


def build_pdf_report(case: pd.Series, analysis: dict[str, Any], contributions: pd.DataFrame) -> bytes:
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=42, leftMargin=42, topMargin=42, bottomMargin=42)
        styles = getSampleStyleSheet()
        story = [
            Paragraph("FraudDetect AI - Rapport d'investigation", styles["Title"]),
            Spacer(1, 14),
            Paragraph(f"Dossier: {case.get('case_id')} | Score: {case.get('risk_score')}/100 | Niveau: {case.get('risk_level')}", styles["Heading2"]),
            Paragraph(str(analysis.get("resume", "")), styles["BodyText"]),
            Spacer(1, 12),
            Paragraph("Justification", styles["Heading2"]),
            Paragraph(str(analysis.get("justification", "")), styles["BodyText"]),
            Spacer(1, 12),
            Paragraph("Actions recommandees", styles["Heading2"]),
        ]
        for action in analysis.get("actions", []):
            story.append(Paragraph(f"- {action}", styles["BodyText"]))
        story.extend([Spacer(1, 12), Paragraph("Top facteurs de risque", styles["Heading2"])])
        table_data = [["Facteur", "Contribution"]] + contributions.head(8).astype(str).values.tolist()
        table = Table(table_data)
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0B132B")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E1")),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("PADDING", (0, 0), (-1, -1), 7),
                ]
            )
        )
        story.append(table)
        doc.build(story)
        return buffer.getvalue()
    except Exception:
        text = [
            "FraudDetect AI - Rapport d'investigation",
            f"Dossier: {case.get('case_id')}",
            f"Score: {case.get('risk_score')}/100",
            f"Niveau: {case.get('risk_level')}",
            "",
            str(analysis.get("resume", "")),
            str(analysis.get("justification", "")),
            "",
            "Actions:",
            *[f"- {x}" for x in analysis.get("actions", [])],
        ]
        return "\n".join(text).encode("utf-8")
