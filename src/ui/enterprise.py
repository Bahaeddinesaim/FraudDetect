from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import PALETTE, RISK_COLORS, RISK_LEVELS


NAV_ICONS = {
    "Accueil": "H",
    "Dashboard executif": "D",
    "Centre d'alertes": "A",
    "Analyse IA": "AI",
    "Investigation": "Q",
    "Portefeuille dossiers": "P",
    "Monitoring modeles": "M",
    "Prevision": "F",
    "Carte fraude": "G",
    "Rapports": "R",
    "Journal d'audit": "J",
    "IA Responsable": "IR",
    "Team de dev": "T",
    "Administration": "AD",
    "Parametres": "S",
}


def inject_enterprise_theme(mode: str) -> None:
    dark = mode == "Dark"
    bg = "#0F172A" if dark else PALETTE["background"]
    card = "#111827" if dark else PALETTE["card"]
    text = "#F8FAFC" if dark else PALETTE["text"]
    muted = "#CBD5E1" if dark else PALETTE["muted"]
    border = "#334155" if dark else PALETTE["border"]
    sidebar = "#0B132B" if dark else "#FFFFFF"
    sidebar_hover = "rgba(255,255,255,.08)" if dark else "#F1F5F9"
    sidebar_text = "#D5DEEA" if dark else "#334155"
    sidebar_arrow = "#94A3B8" if dark else "#64748B"
    st.markdown(
        f"""
        <style>
        :root {{
            --fd-bg:{bg}; --fd-card:{card}; --fd-text:{text}; --fd-muted:{muted};
            --fd-border:{border}; --fd-primary:{PALETTE["primary"]}; --fd-accent:{PALETTE["accent"]};
            --fd-success:{PALETTE["success"]}; --fd-warning:{PALETTE["warning"]}; --fd-danger:{PALETTE["danger"]};
        }}
        .stApp {{ background: var(--fd-bg); color: var(--fd-text); }}
        .block-container {{ max-width: none; padding: 1.2rem 1.35rem 3rem; }}
        h1,h2,h3,h4,p,span,label {{ letter-spacing: 0; }}
        [data-testid="stSidebar"] {{ background:{sidebar}; border-right:1px solid var(--fd-border); }}
        [data-testid="stSidebar"] * {{ letter-spacing:0; }}
        [data-testid="stSidebarContent"] {{ padding: .35rem .78rem 1rem .78rem; }}
        [data-testid="stSidebar"] .stImage {{ margin: 0 0 .48rem 0; padding: 0; }}
        [data-testid="stSidebar"] .stImage img {{
            display: block; max-width: 100%; margin: 0 auto; border-radius: 0;
        }}
        [data-testid="stSidebar"] [role="radiogroup"] {{
            display: flex; flex-direction: column; gap: 6px; margin-top: 0;
        }}
        [data-testid="stSidebar"] [role="radiogroup"] label {{
            position: relative; width: 100%; min-height: 42px; padding: 0 12px !important;
            border-radius: 14px; background: transparent; border: 1px solid transparent;
            transition: background .18s ease, transform .18s ease, box-shadow .18s ease, border-color .18s ease;
        }}
        [data-testid="stSidebar"] [role="radiogroup"] label:hover {{
            background: {sidebar_hover}; transform: translateX(2px);
        }}
        [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {{
            background: linear-gradient(135deg, #3A86FF 0%, #4361EE 100%);
            box-shadow: 0 12px 24px rgba(58, 134, 255, .26);
            border-color: rgba(255,255,255,.16);
        }}
        [data-testid="stSidebar"] [role="radiogroup"] label > div:first-child {{ display: none; }}
        [data-testid="stSidebar"] [role="radiogroup"] label div[data-testid="stMarkdownContainer"] p {{
            display: flex; align-items: center; justify-content: space-between; width: 100%;
            margin: 0; color: {sidebar_text}; font-size: .91rem; font-weight: 740; line-height: 42px;
        }}
        [data-testid="stSidebar"] [role="radiogroup"] label div[data-testid="stMarkdownContainer"] p::after {{
            content: ">"; color: {sidebar_arrow}; font-size: 1rem; line-height: 1;
            transition: color .18s ease, transform .18s ease;
        }}
        [data-testid="stSidebar"] [role="radiogroup"] label:hover div[data-testid="stMarkdownContainer"] p::after {{
            color: var(--fd-accent); transform: translateX(2px);
        }}
        [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) div[data-testid="stMarkdownContainer"] p,
        [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) div[data-testid="stMarkdownContainer"] p::after {{
            color: #FFFFFF !important;
        }}
        [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked):hover {{ transform: translateX(0); }}
        .hero {{
            position: relative; overflow:hidden; border:1px solid rgba(255,255,255,.16);
            background: linear-gradient(135deg, rgba(11,19,43,.96), rgba(28,37,65,.93)), radial-gradient(circle at top right, rgba(58,134,255,.35), transparent 34%);
            color:white; border-radius: 8px; padding: 26px 30px; margin-bottom: 18px;
            box-shadow: 0 22px 55px rgba(15,23,42,.16); backdrop-filter: blur(18px);
        }}
        .hero h1 {{ margin:0 0 8px; color:white; font-size:clamp(2rem,3vw,3rem); font-weight:900; }}
        .hero p {{ color:#DCE6F8; margin:0; max-width: 980px; line-height:1.55; }}
        .badge-row {{ display:flex; gap:8px; flex-wrap:wrap; margin-bottom:14px; }}
        .badge {{
            display:inline-flex; align-items:center; gap:6px; padding:6px 10px; border-radius:999px;
            background:rgba(255,255,255,.12); border:1px solid rgba(255,255,255,.18);
            color:white; font-size:.72rem; font-weight:850; text-transform:uppercase;
        }}
        .kpi {{
            min-height:128px; border-radius:8px; padding:17px; background:var(--fd-card);
            border:1px solid var(--fd-border); box-shadow:0 14px 30px rgba(15,23,42,.07);
            transition: transform .16s ease, box-shadow .16s ease;
        }}
        .kpi:hover {{ transform: translateY(-2px); box-shadow:0 20px 44px rgba(15,23,42,.12); }}
        .kpi-icon {{ width:34px; height:34px; border-radius:8px; display:grid; place-items:center; color:white; font-weight:900; }}
        .kpi-top {{ display:flex; justify-content:space-between; gap:12px; align-items:flex-start; }}
        .kpi-label {{ color:var(--fd-muted); font-size:.76rem; font-weight:850; text-transform:uppercase; }}
        .kpi-value {{ color:var(--fd-text); font-size:1.65rem; line-height:1.1; font-weight:950; margin-top:10px; }}
        .kpi-note {{ color:var(--fd-muted); font-size:.84rem; margin-top:7px; }}
        .panel {{
            background:var(--fd-card); border:1px solid var(--fd-border); border-radius:8px;
            padding:18px; box-shadow:0 14px 30px rgba(15,23,42,.06); height:100%;
        }}
        .panel h3 {{ margin-top:0; color:var(--fd-text); font-size:1.04rem; }}
        .muted {{ color:var(--fd-muted); }}
        .risk-pill {{ display:inline-flex; padding:6px 10px; border-radius:999px; color:white; font-weight:850; font-size:.78rem; }}
        .timeline-item {{ border-left:3px solid var(--fd-accent); padding:0 0 16px 14px; margin-left:6px; }}
        .timeline-time {{ font-weight:950; color:var(--fd-text); }}
        .timeline-detail {{ color:var(--fd-muted); }}
        .team-grid {{ display:grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap:18px; }}
        .team-card {{
            position:relative; overflow:hidden; background:var(--fd-card); border:1px solid var(--fd-border);
            border-radius:12px; padding:22px; box-shadow:0 16px 36px rgba(15,23,42,.08);
            transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease;
        }}
        .team-card:hover {{
            transform:translateY(-3px); box-shadow:0 24px 52px rgba(15,23,42,.13);
            border-color:rgba(58,134,255,.45);
        }}
        .team-card::before {{
            content:""; position:absolute; inset:0 0 auto 0; height:4px;
            background:linear-gradient(90deg, #3A86FF, #4361EE, #06D6A0);
        }}
        .team-avatar {{
            width:56px; height:56px; border-radius:16px; display:grid; place-items:center;
            color:white; font-weight:950; font-size:1rem;
            background:linear-gradient(135deg, #3A86FF 0%, #4361EE 100%);
            box-shadow:0 14px 26px rgba(58,134,255,.25); margin-bottom:16px;
        }}
        .team-name {{ color:var(--fd-text); font-size:1.28rem; line-height:1.15; font-weight:950; margin-bottom:6px; }}
        .team-role {{ color:var(--fd-muted); font-size:.94rem; font-weight:780; margin-bottom:18px; }}
        .team-link {{
            display:inline-flex; align-items:center; justify-content:center; min-height:40px; padding:0 14px;
            border-radius:12px; background:linear-gradient(135deg, #3A86FF 0%, #4361EE 100%);
            color:#FFFFFF !important; text-decoration:none !important; font-weight:900;
            box-shadow:0 12px 22px rgba(58,134,255,.22); transition:transform .18s ease, box-shadow .18s ease;
        }}
        .team-link:hover {{ transform:translateY(-1px); box-shadow:0 16px 30px rgba(58,134,255,.30); }}
        div[data-testid="stDataFrame"] {{ border-radius:8px; overflow:hidden; }}
        .stButton > button {{ border-radius:8px; font-weight:850; border:1px solid var(--fd-accent); background:var(--fd-accent); color:white; }}
        .stButton > button:hover {{ border-color:#2563EB; background:#2563EB; color:white; }}
        @media(max-width: 900px) {{
            .hero {{ padding:20px; }}
            .kpi {{ min-height:auto; }}
            .team-grid {{ grid-template-columns: 1fr; }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero(title: str, subtitle: str, badges: list[str] | None = None) -> None:
    badge_html = "".join(f'<span class="badge">{badge}</span>' for badge in (badges or []))
    st.markdown(
        f"""
        <div class="hero">
            <div class="badge-row">{badge_html}</div>
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def kpi_card(label: str, value: str, note: str, icon: str, color: str = PALETTE["accent"]) -> None:
    st.markdown(
        f"""
        <div class="kpi">
            <div class="kpi-top">
                <div>
                    <div class="kpi-label">{label}</div>
                    <div class="kpi-value">{value}</div>
                </div>
                <div class="kpi-icon" style="background:{color};">{icon}</div>
            </div>
            <div class="kpi-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def panel(title: str, body: str) -> None:
    st.markdown(f'<div class="panel"><h3>{title}</h3><p class="muted">{body}</p></div>', unsafe_allow_html=True)


def apply_chart_theme(fig: go.Figure, height: int = 360) -> go.Figure:
    dark = st.session_state.get("theme_mode") == "Dark"
    text = "#F8FAFC" if dark else PALETTE["text"]
    grid = "#334155" if dark else "#E5E7EB"
    fig.update_layout(
        height=height,
        margin=dict(l=12, r=12, t=48, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=text, family="Inter, Arial, sans-serif"),
        legend=dict(orientation="h", y=1.08, x=1, xanchor="right"),
    )
    fig.update_xaxes(gridcolor=grid, zerolinecolor=grid)
    fig.update_yaxes(gridcolor=grid, zerolinecolor=grid)
    return fig


def render_sidebar(cases: pd.DataFrame, navigation: list[str], gemini_model: str, default_api_key: str) -> dict[str, Any]:
    with st.sidebar:
        if Path("logo.png").exists():
            st.image("logo.png", use_container_width=True)
        if st.session_state.get("active_page") not in navigation:
            st.session_state.active_page = navigation[0]
        page = st.radio(
            "Navigation",
            navigation,
            format_func=lambda item: f"{NAV_ICONS.get(item, '.') }  {item}",
            key="active_page",
            label_visibility="collapsed",
        )
        min_date = cases["submitted_at"].dt.date.min()
        max_date = cases["submitted_at"].dt.date.max()
        return {
            "page": page,
            "api_key": default_api_key,
            "regions": sorted(cases["region"].unique()),
            "risk_levels": RISK_LEVELS,
            "statuses": sorted(cases["status"].unique()),
            "date_range": (min_date, max_date),
        }


def risk_badge(level: str) -> str:
    return f'<span class="risk-pill" style="background:{RISK_COLORS.get(level, PALETTE["muted"])};">{level}</span>'
