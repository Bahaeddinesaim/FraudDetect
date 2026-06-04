from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import PALETTE, RISK_COLORS, RISK_LEVELS


def inject_enterprise_theme(mode: str) -> None:
    dark = mode == "Dark"
    bg = "#0F172A" if dark else PALETTE["background"]
    card = "#111827" if dark else PALETTE["card"]
    text = "#F8FAFC" if dark else PALETTE["text"]
    muted = "#CBD5E1" if dark else PALETTE["muted"]
    border = "#334155" if dark else PALETTE["border"]
    sidebar = "#0B132B" if dark else "#FFFFFF"
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
        .side-card {{ border:1px solid rgba(148,163,184,.25); border-radius:8px; padding:13px; margin:12px 0; background:rgba(148,163,184,.08); }}
        .side-title {{ font-weight:950; color:var(--fd-text); }}
        .side-k {{ color:var(--fd-muted); font-size:.72rem; font-weight:850; text-transform:uppercase; margin-top:8px; }}
        .side-v {{ color:var(--fd-text); font-weight:850; }}
        .timeline-item {{ border-left:3px solid var(--fd-accent); padding:0 0 16px 14px; margin-left:6px; }}
        .timeline-time {{ font-weight:950; color:var(--fd-text); }}
        .timeline-detail {{ color:var(--fd-muted); }}
        div[data-testid="stDataFrame"] {{ border-radius:8px; overflow:hidden; }}
        .stButton > button {{ border-radius:8px; font-weight:850; border:1px solid var(--fd-accent); background:var(--fd-accent); color:white; }}
        .stButton > button:hover {{ border-color:#2563EB; background:#2563EB; color:white; }}
        @media(max-width: 900px) {{ .hero {{ padding:20px; }} .kpi {{ min-height:auto; }} }}
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
        if st.session_state.get("theme_mode") == "Dark":
            logo_bg = "#3A86FF"
        else:
            logo_bg = "#0B132B"
        if Path("logo.png").exists():
            st.markdown(
                """
                <div class="side-card" style="padding:10px;">
                """,
                unsafe_allow_html=True,
            )
            st.image("logo.png", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="side-card">
                <div style="display:flex;gap:12px;align-items:center;">
                    <div style="width:42px;height:42px;border-radius:8px;background:{logo_bg};color:white;display:grid;place-items:center;font-weight:950;">FD</div>
                    <div>
                        <div class="side-title">FraudDetect AI</div>
                        <div class="muted" style="font-size:.78rem;">Fraud Risk Intelligence</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        page = st.radio("Navigation", navigation, label_visibility="collapsed")
        st.divider()
        theme_index = 1 if st.session_state.get("theme_mode") == "Dark" else 0
        st.session_state.theme_mode = st.radio("Theme", ["Light", "Dark"], index=theme_index, horizontal=True)
        api_key = st.text_input("Cle Gemini API", value=default_api_key, type="password")
        st.markdown(
            f"""
            <div class="side-card">
                <div class="side-k">Statut Gemini</div>
                <div class="side-v">{'Connecte' if api_key else 'Fallback local'}</div>
                <div class="side-k">Modele IA</div>
                <div class="side-v">{gemini_model}</div>
                <div class="side-k">Modele ML</div>
                <div class="side-v">Random Forest actif</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.divider()
        st.caption("Filtres globaux")
        regions = st.multiselect("Regions", sorted(cases["region"].unique()), default=sorted(cases["region"].unique()))
        risk_levels = st.multiselect("Niveaux", RISK_LEVELS, default=RISK_LEVELS)
        statuses = st.multiselect("Statuts", sorted(cases["status"].unique()), default=sorted(cases["status"].unique()))
        min_date = cases["submitted_at"].dt.date.min()
        max_date = cases["submitted_at"].dt.date.max()
        date_range = st.date_input("Periode", value=(min_date, max_date), min_value=min_date, max_value=max_date)
        if isinstance(date_range, date):
            date_range = (date_range, date_range)
        st.markdown(
            f"""
            <div class="side-card">
                <div class="side-k">Session</div>
                <div class="side-v">Analyste fraude</div>
                <div class="side-k">Dossiers visibles</div>
                <div class="side-v">{len(cases):,}</div>
            </div>
            """.replace(",", " "),
            unsafe_allow_html=True,
        )
        return {
            "page": page,
            "api_key": api_key,
            "regions": regions,
            "risk_levels": risk_levels,
            "statuses": statuses,
            "date_range": date_range,
        }


def risk_badge(level: str) -> str:
    return f'<span class="risk-pill" style="background:{RISK_COLORS.get(level, PALETTE["muted"])};">{level}</span>'
