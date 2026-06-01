from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


APP_TITLE = "FraudDetect-ai"
GEMINI_MODEL = "gemini-1.5-flash"

COLORS = {
    "night": "#0B1220",
    "night_2": "#111827",
    "green": "#10B981",
    "green_dark": "#047857",
    "blue": "#2563EB",
    "red": "#DC2626",
    "orange": "#F59E0B",
    "bg": "#F4F7FB",
    "card": "#FFFFFF",
    "border": "#E5E7EB",
    "text": "#111827",
    "muted": "#6B7280",
}


@dataclass
class GeminiResult:
    data: dict[str, Any]
    provider: str
    error: str | None = None


def configure_page() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.markdown(
        f"""
        <style>
        :root {{
            --night: {COLORS["night"]};
            --green: {COLORS["green"]};
            --red: {COLORS["red"]};
            --bg: {COLORS["bg"]};
            --card: {COLORS["card"]};
            --border: {COLORS["border"]};
            --text: {COLORS["text"]};
            --muted: {COLORS["muted"]};
        }}

        .stApp {{
            background: var(--bg);
            color: var(--text);
        }}

        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #0B1220 0%, #111827 100%);
            border-right: 1px solid rgba(255,255,255,.08);
        }}

        [data-testid="stSidebar"] * {{
            color: #F9FAFB;
        }}

        .block-container {{
            max-width: 1400px;
            padding-top: 1.6rem;
            padding-bottom: 3rem;
        }}

        h1, h2, h3, h4, p, label, span {{
            letter-spacing: 0;
        }}

        .hero {{
            background: linear-gradient(135deg, #0B1220 0%, #111827 58%, #047857 130%);
            color: white;
            border-radius: 20px;
            padding: 28px 30px;
            margin-bottom: 20px;
            box-shadow: 0 20px 55px rgba(11, 18, 32, .16);
        }}

        .hero h1 {{
            color: white;
            margin: 0 0 8px 0;
            font-size: clamp(2rem, 4vw, 3.1rem);
            line-height: 1.05;
            font-weight: 900;
        }}

        .hero p {{
            color: #D1D5DB;
            max-width: 900px;
            line-height: 1.6;
            margin: 0;
        }}

        .badge {{
            display: inline-flex;
            align-items: center;
            gap: 8px;
            border: 1px solid rgba(255,255,255,.18);
            background: rgba(255,255,255,.10);
            color: white;
            border-radius: 999px;
            padding: 7px 12px;
            font-size: .75rem;
            font-weight: 800;
            text-transform: uppercase;
            margin-bottom: 14px;
        }}

        .brand {{
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 8px 0 20px 0;
        }}

        .brand-mark {{
            width: 44px;
            height: 44px;
            border-radius: 13px;
            background: #10B981;
            color: #052E1C;
            display: grid;
            place-items: center;
            font-weight: 950;
            box-shadow: 0 14px 28px rgba(16,185,129,.22);
        }}

        .brand-title {{
            font-size: 1.05rem;
            font-weight: 900;
            line-height: 1.1;
        }}

        .brand-subtitle {{
            color: #CBD5E1;
            font-size: .78rem;
            margin-top: 3px;
        }}

        .side-card {{
            border: 1px solid rgba(255,255,255,.10);
            background: rgba(255,255,255,.055);
            border-radius: 16px;
            padding: 14px;
            margin: 14px 0;
        }}

        .side-k {{
            color: #94A3B8;
            font-size: .72rem;
            text-transform: uppercase;
            font-weight: 800;
            margin-bottom: 5px;
        }}

        .side-v {{
            color: white;
            font-size: .96rem;
            font-weight: 850;
            margin-bottom: 12px;
        }}

        .kpi-card {{
            background: white;
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 18px;
            min-height: 128px;
            box-shadow: 0 12px 30px rgba(15, 23, 42, .055);
        }}

        .kpi-label {{
            color: var(--muted);
            font-size: .78rem;
            text-transform: uppercase;
            font-weight: 850;
            margin-bottom: 10px;
        }}

        .kpi-value {{
            color: var(--text);
            font-weight: 950;
            font-size: 1.7rem;
            line-height: 1.1;
        }}

        .kpi-note {{
            color: var(--muted);
            font-size: .86rem;
            margin-top: 8px;
        }}

        .panel {{
            background: white;
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 18px;
            box-shadow: 0 12px 30px rgba(15, 23, 42, .05);
            height: 100%;
        }}

        .panel h3 {{
            margin-top: 0;
            color: var(--text);
            font-size: 1.05rem;
        }}

        .ai-box {{
            background: #F8FAFC;
            border: 1px solid #E2E8F0;
            border-left: 5px solid #10B981;
            border-radius: 16px;
            padding: 15px 16px;
            margin: 10px 0;
        }}

        .ai-title {{
            color: #334155;
            text-transform: uppercase;
            font-weight: 900;
            font-size: .75rem;
            margin-bottom: 5px;
        }}

        .ai-text {{
            color: #0F172A;
            line-height: 1.55;
            font-weight: 650;
        }}

        .risk-pill {{
            display: inline-flex;
            border-radius: 999px;
            padding: 7px 11px;
            color: white;
            font-weight: 850;
            font-size: .82rem;
        }}

        .stButton > button {{
            background: #10B981;
            color: #06281B;
            border: 1px solid #10B981;
            border-radius: 13px;
            width: 100%;
            font-weight: 900;
            box-shadow: 0 12px 22px rgba(16,185,129,.18);
        }}

        .stButton > button:hover {{
            background: #059669;
            border-color: #059669;
            color: white;
        }}

        div[data-testid="stMetric"] {{
            background: white;
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 14px;
        }}

        .small-muted {{
            color: #64748B;
            font-size: .88rem;
            line-height: 1.55;
        }}

        @media(max-width: 900px) {{
            .hero {{
                padding: 22px;
            }}
            .kpi-card {{
                min-height: auto;
            }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def make_client_cases(n: int = 420) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    today = date.today()
    channels = ["Mobile App", "Web", "Agence", "Email", "Telephone", "Chat"]
    priorities = ["Basse", "Normale", "Haute", "Critique"]
    segments = ["Retail", "Premium", "Professionnel", "PME", "Private Banking"]
    motifs = [
        "Carte bloquee",
        "Transaction contestee",
        "Virement retarde",
        "Frais bancaires",
        "Acces compte",
        "Credit immobilier",
        "Fraude suspectee",
        "Cloture compte",
    ]

    rows = []
    for idx in range(n):
        priority = rng.choice(priorities, p=[0.18, 0.48, 0.25, 0.09])
        motif = rng.choice(motifs, p=[0.12, 0.18, 0.14, 0.12, 0.13, 0.09, 0.14, 0.08])
        channel = rng.choice(channels, p=[0.26, 0.22, 0.11, 0.14, 0.17, 0.10])
        segment = rng.choice(segments, p=[0.44, 0.21, 0.16, 0.14, 0.05])
        created = today - timedelta(days=int(rng.integers(0, 120)))
        exposure = float(
            rng.lognormal(mean=7.5, sigma=1.0)
            * (1.0 + 0.9 * (priority == "Critique") + 0.45 * (motif == "Fraude suspectee"))
        )
        sla_hours = int(rng.integers(1, 96))
        risk_score = int(
            np.clip(
                rng.normal(35, 18)
                + 20 * (priority == "Haute")
                + 37 * (priority == "Critique")
                + 16 * (motif in ["Fraude suspectee", "Transaction contestee"])
                + 6 * (sla_hours > 48),
                0,
                100,
            )
        )
        rows.append(
            {
                "case_id": f"NT-{2026}-{idx + 1001}",
                "date": pd.Timestamp(created),
                "week": pd.Timestamp(created).to_period("W").start_time,
                "channel": channel,
                "priority": priority,
                "segment": segment,
                "motif": motif,
                "customer": f"Client {idx + 1:03d}",
                "sla_hours": sla_hours,
                "risk_score": risk_score,
                "financial_exposure": round(exposure, 2),
                "status": rng.choice(["Nouveau", "En analyse", "En attente client", "Resolue"], p=[0.28, 0.36, 0.18, 0.18]),
            }
        )
    return pd.DataFrame(rows)


def filter_cases(df: pd.DataFrame, channels: list[str], priorities: list[str], segments: list[str]) -> pd.DataFrame:
    return df[
        df["channel"].isin(channels)
        & df["priority"].isin(priorities)
        & df["segment"].isin(segments)
    ].copy()


def kpi_card(label: str, value: str, note: str) -> None:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def chart_layout(fig: go.Figure, height: int = 340) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=45, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLORS["text"], family="Inter, Arial, sans-serif"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def priority_color(priority: str) -> str:
    return {
        "Basse": COLORS["green"],
        "Normale": COLORS["blue"],
        "Haute": COLORS["orange"],
        "Critique": COLORS["red"],
    }.get(priority, COLORS["muted"])


def safe_json_extract(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if match:
            return json.loads(match.group(0))
        raise


def fallback_analysis(message: str, client_segment: str, channel: str, priority_hint: str) -> dict[str, Any]:
    lowered = message.lower()
    fraud_terms = ["fraude", "pirate", "vol", "carte", "transaction inconnue", "opposition"]
    anger_terms = ["urgent", "inadmissible", "plainte", "bloque", "impossible", "remboursement"]
    risk = 35
    risk += 35 if any(term in lowered for term in fraud_terms) else 0
    risk += 18 if any(term in lowered for term in anger_terms) else 0
    risk += {"Basse": 0, "Normale": 8, "Haute": 20, "Critique": 34}.get(priority_hint, 8)
    risk += 7 if client_segment in ["Private Banking", "PME"] else 0
    risk = int(np.clip(risk, 0, 100))

    urgency = "Critique" if risk >= 75 else "Haute" if risk >= 55 else "Normale" if risk >= 30 else "Basse"
    sentiment = "Negatif" if any(term in lowered for term in anger_terms) else "Neutre"
    category = "Fraude suspectee" if any(term in lowered for term in fraud_terms) else "Service client"
    escalate = risk >= 70

    return {
        "resume_operationnel": "Le client signale une situation necessitant une qualification rapide par le support bancaire.",
        "intention_client": "Obtenir une prise en charge et une reponse claire de la banque.",
        "sentiment_client": sentiment,
        "niveau_urgence": urgency,
        "score_risque": risk,
        "categorie_reclamation": category,
        "reponse_proposee": (
            "Bonjour, nous avons bien recu votre demande. Votre dossier est pris en charge par notre equipe. "
            "Par securite, nous allons verifier les operations concernees et revenir vers vous avec les prochaines etapes."
        ),
        "actions_recommandees": [
            "Verifier l'identite du client selon la procedure KYC.",
            "Consulter l'historique recent du compte et les operations litigieuses.",
            "Ouvrir une investigation interne si une fraude est suspectee.",
            "Informer le client des delais de traitement et des mesures de protection.",
        ],
        "justification_decision": "Le score combine les mots-clés du message, la priorite indiquee, le canal et le segment client.",
        "niveau_confiance": 0.72,
        "escalade_necessaire": escalate,
        "resume_superviseur": "Dossier a suivre avec priorisation adaptee au risque et controle humain avant toute decision.",
    }


def build_gemini_prompt(message: str, client_segment: str, channel: str, priority_hint: str) -> str:
    return f"""
Tu es un agent IA bancaire pour FraudDetect-ai. Analyse le message client ci-dessous.
Retourne uniquement un JSON valide, sans markdown, avec exactement les cles suivantes:
resume_operationnel, intention_client, sentiment_client, niveau_urgence, score_risque,
categorie_reclamation, reponse_proposee, actions_recommandees, justification_decision,
niveau_confiance, escalade_necessaire, resume_superviseur.

Contraintes:
- score_risque est un entier de 0 a 100.
- niveau_confiance est un nombre entre 0 et 1.
- escalade_necessaire est un booleen.
- actions_recommandees est une liste de chaines.
- Ne prends jamais de decision automatique de sanction, cloture ou remboursement definitif.
- Reste professionnel, prudent, conforme au human-in-the-loop.

Contexte dossier:
- Segment client: {client_segment}
- Canal: {channel}
- Priorite initiale: {priority_hint}

Message client:
{message}
"""


def analyze_with_gemini(api_key: str, message: str, client_segment: str, channel: str, priority_hint: str) -> GeminiResult:
    if not api_key:
        return GeminiResult(
            data=fallback_analysis(message, client_segment, channel, priority_hint),
            provider="Fallback simule",
        )
    try:
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(
            build_gemini_prompt(message, client_segment, channel, priority_hint),
            generation_config={"temperature": 0.2, "response_mime_type": "application/json"},
        )
        return GeminiResult(data=safe_json_extract(response.text), provider=f"Gemini API - {GEMINI_MODEL}")
    except Exception as exc:
        return GeminiResult(
            data=fallback_analysis(message, client_segment, channel, priority_hint),
            provider="Fallback simule",
            error=str(exc),
        )


def render_ai_summary(result: dict[str, Any]) -> None:
    risk = int(result.get("score_risque", 0))
    color = COLORS["green"] if risk < 40 else COLORS["orange"] if risk < 70 else COLORS["red"]
    st.markdown(
        f"""
        <div class="panel">
            <div style="display:flex;justify-content:space-between;gap:14px;align-items:flex-start;flex-wrap:wrap;">
                <div>
                    <h3>Analyse IA du dossier</h3>
                    <p class="small-muted">Synthese operationnelle et recommandations pour l'agent.</p>
                </div>
                <div class="risk-pill" style="background:{color};">Risque {risk}/100</div>
            </div>
            <div class="ai-box">
                <div class="ai-title">Resume operationnel</div>
                <div class="ai-text">{result.get("resume_operationnel", "-")}</div>
            </div>
            <div class="ai-box">
                <div class="ai-title">Reponse proposee au client</div>
                <div class="ai-text">{result.get("reponse_proposee", "-")}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Intention", result.get("intention_client", "-"))
    col2.metric("Sentiment", result.get("sentiment_client", "-"))
    col3.metric("Urgence", result.get("niveau_urgence", "-"))
    col4.metric("Escalade", "Oui" if result.get("escalade_necessaire") else "Non")

    action_col, why_col = st.columns([0.52, 0.48], gap="large")
    with action_col:
        st.markdown("#### Actions recommandees")
        for action in result.get("actions_recommandees", []):
            st.markdown(f"- {action}")
    with why_col:
        st.markdown("#### Justification")
        st.write(result.get("justification_decision", "-"))
        st.metric("Niveau de confiance", f"{float(result.get('niveau_confiance', 0)):.0%}")
        st.markdown("#### Resume superviseur")
        st.write(result.get("resume_superviseur", "-"))


def dashboard_page(df: pd.DataFrame) -> None:
    st.markdown(
        """
        <div class="hero">
            <div class="badge">Executive Dashboard</div>
            <h1>FraudDetect-ai</h1>
            <p>Vue executive des reclamations clients, risques SLA, exposition financiere et priorisation operationnelle.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    critical = df[df["priority"] == "Critique"]
    sla_risk = float((df["sla_hours"] > 48).mean() * 100) if len(df) else 0
    total_exposure = df["financial_exposure"].sum()
    avg_risk = df["risk_score"].mean() if len(df) else 0

    kpis = st.columns(4)
    with kpis[0]:
        kpi_card("Dossiers ouverts", f"{len(df):,}".replace(",", " "), "Volume filtre")
    with kpis[1]:
        kpi_card("Cas critiques", f"{len(critical):,}".replace(",", " "), "Priorite maximale")
    with kpis[2]:
        kpi_card("Exposition", f"{total_exposure/1_000_000:.2f} M€", "Montant potentiel")
    with kpis[3]:
        kpi_card("Risque moyen", f"{avg_risk:.0f}/100", "Score operationnel")

    row1_a, row1_b = st.columns([0.48, 0.52], gap="large")
    with row1_a:
        priority_counts = df["priority"].value_counts().reindex(["Basse", "Normale", "Haute", "Critique"]).dropna()
        fig = px.bar(
            priority_counts,
            x=priority_counts.index,
            y=priority_counts.values,
            title="Volume des demandes par priorite",
            color=priority_counts.index,
            color_discrete_map={p: priority_color(p) for p in priority_counts.index},
            labels={"x": "Priorite", "y": "Dossiers"},
        )
        st.plotly_chart(chart_layout(fig), use_container_width=True)
    with row1_b:
        channel_counts = df["channel"].value_counts()
        fig = px.pie(
            channel_counts,
            names=channel_counts.index,
            values=channel_counts.values,
            hole=0.55,
            title="Repartition des canaux de contact",
            color_discrete_sequence=[COLORS["green"], COLORS["blue"], COLORS["orange"], "#64748B", "#7C3AED", COLORS["red"]],
        )
        st.plotly_chart(chart_layout(fig), use_container_width=True)

    row2_a, row2_b = st.columns([0.56, 0.44], gap="large")
    with row2_a:
        timeline = df.groupby("week", as_index=False).agg(dossiers=("case_id", "count"))
        fig = px.line(timeline, x="week", y="dossiers", markers=True, title="Evolution temporelle des dossiers")
        fig.update_traces(line=dict(color=COLORS["green"], width=3), marker=dict(size=7))
        st.plotly_chart(chart_layout(fig), use_container_width=True)
    with row2_b:
        critical_week = critical.groupby("week", as_index=False).agg(critiques=("case_id", "count"))
        fig = px.bar(critical_week, x="week", y="critiques", title="Cas critiques par semaine")
        fig.update_traces(marker_color=COLORS["red"])
        st.plotly_chart(chart_layout(fig), use_container_width=True)

    row3_a, row3_b = st.columns([0.5, 0.5], gap="large")
    with row3_a:
        seg = df.groupby("segment", as_index=False).agg(exposition=("financial_exposure", "sum"))
        fig = px.bar(seg.sort_values("exposition"), x="exposition", y="segment", orientation="h", title="Exposition financiere potentielle")
        fig.update_traces(marker_color=COLORS["green"])
        st.plotly_chart(chart_layout(fig), use_container_width=True)
    with row3_b:
        segment_counts = df["segment"].value_counts()
        fig = px.bar(
            x=segment_counts.index,
            y=segment_counts.values,
            title="Repartition par segment client",
            labels={"x": "Segment", "y": "Dossiers"},
        )
        fig.update_traces(marker_color=COLORS["blue"])
        st.plotly_chart(chart_layout(fig), use_container_width=True)

    row4_a, row4_b = st.columns([0.5, 0.5], gap="large")
    with row4_a:
        motifs = df["motif"].value_counts().head(8).sort_values()
        fig = px.bar(x=motifs.values, y=motifs.index, orientation="h", title="Top motifs de reclamation")
        fig.update_traces(marker_color=COLORS["night"])
        st.plotly_chart(chart_layout(fig), use_container_width=True)
    with row4_b:
        heat = pd.crosstab(df["priority"], df["channel"]).reindex(["Basse", "Normale", "Haute", "Critique"]).fillna(0)
        fig = px.imshow(
            heat,
            text_auto=True,
            aspect="auto",
            title="Heatmap priorite x canal",
            color_continuous_scale=["#ECFDF5", "#10B981", "#0B1220"],
        )
        st.plotly_chart(chart_layout(fig), use_container_width=True)

    row5_a, row5_b = st.columns([0.36, 0.64], gap="large")
    with row5_a:
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=sla_risk,
                title={"text": "Gauge du risque SLA"},
                number={"suffix": "%"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": COLORS["red"] if sla_risk >= 35 else COLORS["orange"] if sla_risk >= 20 else COLORS["green"]},
                    "steps": [
                        {"range": [0, 20], "color": "#DCFCE7"},
                        {"range": [20, 35], "color": "#FEF3C7"},
                        {"range": [35, 100], "color": "#FEE2E2"},
                    ],
                },
            )
        )
        st.plotly_chart(chart_layout(fig, height=330), use_container_width=True)
    with row5_b:
        st.markdown("#### Dossiers les plus critiques")
        critical_table = df.sort_values(["risk_score", "financial_exposure"], ascending=False).head(12)
        st.dataframe(
            critical_table[
                [
                    "case_id",
                    "customer",
                    "priority",
                    "channel",
                    "segment",
                    "motif",
                    "risk_score",
                    "financial_exposure",
                    "sla_hours",
                    "status",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )


def analysis_page(api_key: str) -> None:
    st.markdown(
        """
        <div class="hero">
            <div class="badge">Gemini Agent</div>
            <h1>Analyse Gemini</h1>
            <p>Agent IA bancaire pour classifier un message client, generer une reponse et produire un resume superviseur.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([0.42, 0.58], gap="large")
    with left:
        with st.container(border=True):
            st.markdown("### Message client")
            segment = st.selectbox("Segment client", ["Retail", "Premium", "Professionnel", "PME", "Private Banking"])
            channel = st.selectbox("Canal", ["Mobile App", "Web", "Agence", "Email", "Telephone", "Chat"])
            priority = st.selectbox("Priorite initiale", ["Basse", "Normale", "Haute", "Critique"], index=1)
            message = st.text_area(
                "Message a analyser",
                height=230,
                value=(
                    "Bonjour, je vois une transaction inconnue de 1280 euros sur ma carte. "
                    "Je n'arrive pas a joindre le service client et je veux bloquer cette operation rapidement."
                ),
            )
            run = st.button("Analyser avec l'agent IA")

    if "last_analysis" not in st.session_state or run:
        st.session_state.last_analysis = analyze_with_gemini(api_key, message, segment, channel, priority)

    result: GeminiResult = st.session_state.last_analysis
    with right:
        if result.error:
            st.warning(f"Gemini indisponible, fallback simule utilise. Detail: {result.error}")
        st.caption(f"Fournisseur: {result.provider}")
        render_ai_summary(result.data)

    st.markdown("### Sortie JSON structuree")
    st.json(result.data)


def responsible_ai_page() -> None:
    st.markdown(
        """
        <div class="hero">
            <div class="badge">Responsible AI</div>
            <h1>Cadrage IA</h1>
            <p>Cadre d'usage responsable pour un assistant bancaire d'aide a la decision.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    sections = [
        ("Objectif du systeme", "Aider les conseillers et superviseurs a analyser rapidement les messages clients, prioriser les dossiers et standardiser les reponses."),
        ("Ce que l'IA peut faire", "Resumer, classifier, proposer des actions internes, rediger une reponse, signaler les risques et recommander une escalade."),
        ("Ce que l'IA ne doit pas faire", "Valider un remboursement, refuser une reclamation, bloquer un compte ou prendre une decision juridique sans controle humain."),
        ("Limites du modele", "Risque d'erreur, dependance au contexte fourni, sensibilite aux formulations ambiguës, absence d'acces direct au SI bancaire."),
        ("Human-in-the-loop", "L'agent humain controle la recommandation, verifie les donnees internes et prend la decision finale."),
        ("Non-decision automatique", "Toute action impactant le client doit etre validee dans les workflows bancaires existants."),
        ("Tracabilite", "Conserver message, sortie structuree, version du prompt, fournisseur IA, horodatage et identifiant agent."),
    ]
    cols = st.columns(2, gap="large")
    for idx, (title, text) in enumerate(sections):
        with cols[idx % 2]:
            st.markdown(
                f"""
                <div class="panel">
                    <h3>{title}</h3>
                    <p class="small-muted">{text}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


def deployment_page() -> None:
    st.markdown(
        """
        <div class="hero">
            <div class="badge">Delivery & MLOps</div>
            <h1>Deploiement</h1>
            <p>Architecture technique, flux de donnees, securite API et trajectoire d'industrialisation.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("### Architecture technique")
    st.code(
        """
Client message -> Streamlit UI -> Gemini FraudDetect-ai Agent
                -> Structured JSON -> Agent dashboard
                -> Human validation -> CRM / ticketing system

Executive data -> Simulated portfolio -> Pandas transforms -> Plotly dashboards
        """,
        language="text",
    )

    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown("### Stack utilisee")
        st.markdown(
            "- Streamlit pour l'interface\n"
            "- Plotly pour les visualisations executives\n"
            "- Pandas / NumPy pour les donnees simulees\n"
            "- Gemini API via `google-generativeai`\n"
            "- Fallback local si aucune cle API n'est fournie"
        )
        st.markdown("### Deploiement local")
        st.code("pip install -r requirements.txt\nstreamlit run app.py", language="bash")
    with col2:
        st.markdown("### Securite API")
        st.markdown(
            "- Cle saisie via champ password dans la sidebar\n"
            "- Cle jamais affichee dans l'interface\n"
            "- Variable d'environnement possible: `GEMINI_API_KEY`\n"
            "- Gestion des erreurs et fallback simule\n"
            "- Aucune donnee bancaire reelle dans cette demonstration"
        )
        st.markdown("### Streamlit Cloud")
        st.code(
            "Repository: Bahaeddinesaim/FraudDetect\nBranch: main\nMain file path: app.py\nSecret: GEMINI_API_KEY",
            language="text",
        )

    st.markdown("### Limites actuelles et ameliorations futures")
    st.markdown(
        "- Donnees simulees, sans connexion CRM reelle\n"
        "- Pas de journalisation persistante des prompts et sorties\n"
        "- Pas de gestion fine des habilitations agent/superviseur\n"
        "- Future version: RAG sur procedures internes, audit trail, evaluation qualite, tests de biais, connecteurs CRM"
    )


def sidebar_controls(df: pd.DataFrame) -> tuple[str, str, list[str], list[str], list[str]]:
    with st.sidebar:
        st.markdown(
            """
            """,
            unsafe_allow_html=True,
        )
        if Path("logo.png").exists():
            st.image("logo.png", use_container_width=True)
        st.markdown(
            """
            <div class="brand">
                <div>
                    <div class="brand-title">FraudDetect-ai</div>
                    <div class="brand-subtitle">Fraud Risk Intelligence</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        page = st.radio(
            "Navigation",
            ["Dashboard executif", "Analyse Gemini", "Cadrage IA", "Deploiement"],
        )
        st.divider()
        api_key = st.text_input(
            "Cle Gemini API",
            value=os.getenv("GEMINI_API_KEY", ""),
            type="password",
            help="La cle reste masquee. Si elle est vide, l'application utilise un fallback simule.",
        )
        st.markdown(
            f"""
            <div class="side-card">
                <div class="side-k">Mode IA</div>
                <div class="side-v">{'Gemini API' if api_key else 'Fallback simule'}</div>
                <div class="side-k">Modele cible</div>
                <div class="side-v">{GEMINI_MODEL}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.divider()
        st.markdown("### Filtres dashboard")
        channels = st.multiselect("Canal", sorted(df["channel"].unique()), default=sorted(df["channel"].unique()))
        priorities = st.multiselect("Priorite", ["Basse", "Normale", "Haute", "Critique"], default=["Basse", "Normale", "Haute", "Critique"])
        segments = st.multiselect("Segment", sorted(df["segment"].unique()), default=sorted(df["segment"].unique()))
        return page, api_key, channels, priorities, segments


def main() -> None:
    configure_page()
    df = make_client_cases()
    page, api_key, channels, priorities, segments = sidebar_controls(df)
    filtered = filter_cases(df, channels, priorities, segments)

    if page == "Dashboard executif":
        dashboard_page(filtered)
    elif page == "Analyse Gemini":
        analysis_page(api_key)
    elif page == "Cadrage IA":
        responsible_ai_page()
    elif page == "Deploiement":
        deployment_page()


if __name__ == "__main__":
    main()
