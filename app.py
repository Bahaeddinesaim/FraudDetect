from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from src.fraudai_core import (
    FEATURES,
    load_bundle,
    load_transactions,
    predict_proba_positive,
    risk_level,
    save_bundle,
    split_xy,
    train_and_select,
)


MODEL_PATH = Path("models/fraudai_model.joblib")


@st.cache_resource
def get_bundle():
    if MODEL_PATH.exists():
        return load_bundle(MODEL_PATH)
    df = load_transactions()
    model, report, _, _, _ = train_and_select(df)
    X, _ = split_xy(df)
    save_bundle(MODEL_PATH, model, report, list(X.columns))
    return {"model": model, "report": report, "columns": list(X.columns)}


def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #F7F8FA;
            --text: #111827;
            --muted: #6B7280;
            --sidebar: #111827;
            --accent: #E30613;
            --card: #FFFFFF;
            --border: #E5E7EB;
            --success: #16A34A;
            --warning: #F59E0B;
            --danger: #DC2626;
        }

        .stApp {
            background: var(--bg);
            color: var(--text);
        }

        [data-testid="stSidebar"] {
            background: var(--sidebar);
            border-right: 1px solid rgba(255, 255, 255, 0.08);
        }

        [data-testid="stSidebar"] * {
            color: #F9FAFB;
        }

        [data-testid="stSidebar"] .stRadio label {
            color: #F9FAFB !important;
        }

        [data-testid="stSidebar"] [role="radiogroup"] label {
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 12px;
            padding: 8px 10px;
            margin-bottom: 6px;
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1320px;
        }

        h1, h2, h3, h4, p, span, label {
            letter-spacing: 0;
        }

        .brand-wrap {
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 4px 0 20px 0;
        }

        .brand-logo {
            width: 42px;
            height: 42px;
            border-radius: 10px;
            background: #E30613;
            color: white;
            display: grid;
            place-items: center;
            font-weight: 800;
            font-size: 15px;
            box-shadow: 0 14px 28px rgba(227, 6, 19, 0.28);
        }

        .brand-title {
            font-size: 1.08rem;
            font-weight: 800;
            line-height: 1.1;
        }

        .brand-subtitle {
            color: #D1D5DB;
            font-size: 0.78rem;
            margin-top: 3px;
        }

        .side-card {
            background: rgba(255, 255, 255, 0.06);
            border: 1px solid rgba(255, 255, 255, 0.10);
            border-radius: 14px;
            padding: 14px;
            margin: 14px 0;
        }

        .side-label {
            color: #9CA3AF;
            font-size: 0.73rem;
            text-transform: uppercase;
            font-weight: 700;
            margin-bottom: 4px;
        }

        .side-value {
            color: white;
            font-size: 0.98rem;
            font-weight: 750;
            margin-bottom: 10px;
        }

        .hero {
            background: linear-gradient(135deg, #111827 0%, #1F2937 58%, #E30613 140%);
            border-radius: 18px;
            padding: 28px 30px;
            color: white;
            border: 1px solid rgba(17, 24, 39, 0.08);
            box-shadow: 0 18px 45px rgba(17, 24, 39, 0.12);
            margin-bottom: 22px;
        }

        .hero-topline {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 16px;
            flex-wrap: wrap;
            margin-bottom: 14px;
        }

        .badge {
            display: inline-flex;
            align-items: center;
            gap: 7px;
            padding: 7px 11px;
            border-radius: 999px;
            font-size: 0.75rem;
            font-weight: 800;
            text-transform: uppercase;
            border: 1px solid rgba(255, 255, 255, 0.22);
            background: rgba(255, 255, 255, 0.11);
            color: #FFFFFF;
            white-space: nowrap;
        }

        .hero h1 {
            color: white;
            font-size: clamp(2rem, 4vw, 3rem);
            line-height: 1.05;
            margin: 0 0 10px 0;
            letter-spacing: 0;
        }

        .hero p {
            color: #E5E7EB;
            max-width: 880px;
            margin: 0;
            font-size: 1.02rem;
            line-height: 1.65;
        }

        .section-title {
            font-size: 1.1rem;
            font-weight: 850;
            margin: 6px 0 12px 0;
            color: var(--text);
        }

        .card {
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 18px;
            box-shadow: 0 10px 28px rgba(17, 24, 39, 0.06);
            height: 100%;
        }

        .kpi-card {
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 17px 18px;
            box-shadow: 0 10px 24px rgba(17, 24, 39, 0.055);
            min-height: 120px;
        }

        .kpi-top {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 10px;
            margin-bottom: 12px;
        }

        .kpi-icon {
            width: 36px;
            height: 36px;
            border-radius: 10px;
            display: grid;
            place-items: center;
            background: #FEF2F2;
            color: var(--accent);
            font-weight: 900;
            font-size: 0.92rem;
        }

        .kpi-label {
            color: var(--muted);
            font-size: 0.78rem;
            text-transform: uppercase;
            font-weight: 800;
        }

        .kpi-value {
            color: var(--text);
            font-size: 1.45rem;
            font-weight: 900;
            line-height: 1.1;
        }

        .kpi-help {
            color: var(--muted);
            font-size: 0.82rem;
            margin-top: 8px;
        }

        .form-note {
            background: #F9FAFB;
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 12px;
            color: var(--muted);
            font-size: 0.88rem;
            line-height: 1.55;
        }

        div[data-testid="stNumberInput"] input,
        div[data-testid="stSelectbox"] div,
        div[data-testid="stSlider"] {
            border-radius: 10px;
        }

        .stButton > button {
            background: var(--accent);
            color: white;
            border: 1px solid var(--accent);
            border-radius: 12px;
            padding: 0.65rem 1rem;
            font-weight: 800;
            width: 100%;
            box-shadow: 0 12px 22px rgba(227, 6, 19, 0.18);
        }

        .stButton > button:hover {
            background: #B9040F;
            border-color: #B9040F;
            color: white;
        }

        .gauge-shell {
            background: #F3F4F6;
            border-radius: 999px;
            height: 18px;
            overflow: hidden;
            border: 1px solid var(--border);
            margin: 16px 0 8px 0;
        }

        .gauge-fill {
            height: 100%;
            border-radius: 999px;
            background: linear-gradient(90deg, #16A34A 0%, #F59E0B 55%, #DC2626 100%);
        }

        .risk-score {
            font-size: clamp(2.6rem, 7vw, 4.8rem);
            line-height: 0.95;
            color: var(--text);
            font-weight: 950;
            letter-spacing: 0;
        }

        .risk-score span {
            font-size: 1.05rem;
            color: var(--muted);
            font-weight: 850;
        }

        .risk-badge {
            display: inline-flex;
            padding: 8px 12px;
            border-radius: 999px;
            color: white;
            font-weight: 850;
            font-size: 0.86rem;
        }

        .action-box {
            background: #F9FAFB;
            border: 1px solid var(--border);
            border-left: 5px solid var(--accent);
            border-radius: 14px;
            padding: 14px 15px;
            margin-top: 14px;
        }

        .action-title {
            font-size: 0.78rem;
            color: var(--muted);
            font-weight: 850;
            text-transform: uppercase;
            margin-bottom: 4px;
        }

        .action-text {
            color: var(--text);
            font-size: 1rem;
            font-weight: 800;
        }

        .explain-row {
            display: grid;
            grid-template-columns: minmax(88px, 0.7fr) 2fr minmax(70px, 0.55fr);
            gap: 12px;
            align-items: center;
            padding: 11px 0;
            border-bottom: 1px solid var(--border);
        }

        .explain-row:last-child {
            border-bottom: 0;
        }

        .feature-name {
            color: var(--text);
            font-weight: 850;
        }

        .impact-track {
            height: 8px;
            background: #F3F4F6;
            border-radius: 999px;
            overflow: hidden;
        }

        .impact-fill {
            height: 100%;
            border-radius: 999px;
            background: var(--accent);
        }

        .impact-value {
            color: var(--muted);
            text-align: right;
            font-size: 0.84rem;
            font-weight: 750;
        }

        .human-card {
            background: #111827;
            color: white;
            border-radius: 16px;
            padding: 18px;
            border: 1px solid #1F2937;
            box-shadow: 0 14px 30px rgba(17, 24, 39, 0.16);
        }

        .human-card h3 {
            color: white;
            margin: 0 0 8px 0;
            font-size: 1.08rem;
        }

        .human-card p {
            color: #D1D5DB;
            margin: 0;
            line-height: 1.6;
        }

        .dataframe {
            border-radius: 12px;
            overflow: hidden;
        }

        @media (max-width: 900px) {
            .hero {
                padding: 22px;
            }
            .explain-row {
                grid-template-columns: 1fr;
                gap: 7px;
            }
            .impact-value {
                text-align: left;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, helper: str, icon: str) -> None:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-top">
                <div class="kpi-label">{label}</div>
                <div class="kpi-icon">{icon}</div>
            </div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-help">{helper}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def normalize_label(label: str) -> str:
    mapping = {"Faible": "Faible", "Modere": "Modere", "Eleve": "Eleve"}
    return mapping.get(label, label)


def risk_color(score: float) -> str:
    if score <= 0.30:
        return "#16A34A"
    if score <= 0.70:
        return "#F59E0B"
    return "#DC2626"


def model_importance(model: Any, columns: list[str], case: pd.DataFrame) -> pd.DataFrame:
    estimator = model
    if hasattr(model, "named_steps"):
        estimator = model.named_steps.get("model", model)

    if hasattr(estimator, "feature_importances_"):
        importance = np.asarray(estimator.feature_importances_, dtype=float)
    elif hasattr(estimator, "coef_"):
        importance = np.abs(np.asarray(estimator.coef_).ravel())
    else:
        values = np.abs(case.iloc[0].to_numpy(dtype=float))
        importance = values / max(values.sum(), 1e-9)

    if len(importance) != len(columns):
        importance = np.resize(importance, len(columns))

    df = pd.DataFrame(
        {
            "variable": columns,
            "importance": importance,
            "valeur_dossier": case.iloc[0].to_numpy(dtype=float),
        }
    )
    df["impact_local"] = df["importance"] * np.abs(df["valeur_dossier"])
    if df["impact_local"].max() <= 0:
        df["impact_local"] = df["importance"]
    df = df.sort_values("impact_local", ascending=False).head(7)
    total = max(df["impact_local"].max(), 1e-9)
    df["impact_pct"] = df["impact_local"] / total
    return df


def render_explainability(importance_df: pd.DataFrame) -> None:
    rows = []
    for _, row in importance_df.iterrows():
        width = int(max(row["impact_pct"] * 100, 4))
        rows.append(
            f"""
            <div class="explain-row">
                <div class="feature-name">{row["variable"]}</div>
                <div class="impact-track"><div class="impact-fill" style="width:{width}%"></div></div>
                <div class="impact-value">{row["valeur_dossier"]:.2f}</div>
            </div>
            """
        )
    st.markdown("".join(rows), unsafe_allow_html=True)


st.set_page_config(page_title="FrauDetectAI", page_icon="FAI", layout="wide")
inject_css()

bundle = get_bundle()
model = bundle["model"]
report = bundle["report"]
columns = bundle["columns"]

with st.sidebar:
    st.markdown(
        """
        <div class="brand-wrap">
            <div class="brand-logo">FAI</div>
            <div>
                <div class="brand-title">FrauDetectAI</div>
                <div class="brand-subtitle">EPITA Fraud Risk Lab</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    page = st.radio(
        "Navigation",
        ["Scoring agent", "Modele ML", "Gouvernance"],
        label_visibility="collapsed",
    )
    st.markdown(
        f"""
        <div class="side-card">
            <div class="side-label">Modele en production</div>
            <div class="side-value">{report.name}</div>
            <div class="side-label">AUC-PR</div>
            <div class="side-value">{report.average_precision:.3f}</div>
            <div class="side-label">Seuil operationnel</div>
            <div class="side-value">{report.threshold:.2f}</div>
        </div>
        <div class="side-card">
            <div class="side-label">Role de l'IA</div>
            <div class="side-value">Priorisation des controles</div>
            <div class="side-label">Decision finale</div>
            <div class="side-value">Agent public</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <section class="hero">
        <div class="hero-topline">
            <div class="badge">AI Risk Scoring</div>
            <div class="badge">Services publics</div>
        </div>
        <h1>FrauDetectAI Dashboard</h1>
        <p>
            Tableau de bord d'aide a la decision pour prioriser les controles de fraude.
            Le modele estime un niveau de risque, explique les signaux principaux et laisse
            la decision finale a l'agent public.
        </p>
    </section>
    """,
    unsafe_allow_html=True,
)

kpi_cols = st.columns(4)
with kpi_cols[0]:
    metric_card("Precision", f"{report.precision:.3f}", "Fiabilite des alertes emises", "P")
with kpi_cols[1]:
    metric_card("Recall", f"{report.recall:.3f}", "Fraudes detectees parmi les cas reels", "R")
with kpi_cols[2]:
    metric_card("F1-score", f"{report.f1:.3f}", "Equilibre precision / recall", "F1")
with kpi_cols[3]:
    metric_card("Faux positifs", f"{report.false_positive_rate:.3%}", "Objectif metier: rester faible", "FP")


def make_case(amount: float, hour: int, scenario: str) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    row = {feature: 0.0 for feature in FEATURES}
    row["Amount"] = amount
    row["Time"] = hour * 3600
    row.update({f"V{i}": rng.normal(0, 0.45) for i in range(1, 29)})
    if scenario == "Montant atypique":
        row["V1"], row["V7"], row["V14"] = 1.4, 1.3, -1.4
    elif scenario == "Signaux incoherents":
        row["V3"], row["V10"], row["V12"] = -1.6, -2.1, -1.7
    elif scenario == "Cas fortement suspect":
        row["V1"], row["V3"], row["V10"], row["V12"], row["V14"], row["V17"] = (
            2.2,
            -2.0,
            -2.7,
            -2.2,
            -3.0,
            -1.6,
        )
    df = pd.DataFrame([row])
    df["Hour"] = (df["Time"] // 3600) % 24
    df["LogAmount"] = np.log1p(df["Amount"])
    df["IsNight"] = df["Hour"].between(0, 5).astype(int)
    return df[columns]


if page == "Scoring agent":
    st.markdown('<div class="section-title">Analyse d\'un dossier</div>', unsafe_allow_html=True)
    left, right = st.columns([0.42, 0.58], gap="large")

    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### Saisie agent")
        with st.form("agent_case_form"):
            amount = st.number_input(
                "Montant de la demande ou transaction",
                min_value=0.0,
                value=128.0,
                step=10.0,
                help="Montant financier associe au dossier a controler.",
            )
            hour = st.slider(
                "Heure de depot",
                min_value=0,
                max_value=23,
                value=14,
                help="Heure de reception ou d'enregistrement du dossier.",
            )
            scenario = st.selectbox(
                "Profil de dossier",
                ["Standard", "Montant atypique", "Signaux incoherents", "Cas fortement suspect"],
                help="Scenario de demonstration qui modifie les variables anonymisees V1 a V28.",
            )
            st.form_submit_button("Calculer le score de risque")

        st.markdown(
            """
            <div class="form-note">
                Les variables V1 a V28 sont des composantes anonymisees issues du dataset de reference.
                En production, elles correspondraient a des signaux metier pseudonymises et documentes.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    case = make_case(amount, hour, scenario)
    score = float(predict_proba_positive(model, case)[0])
    label, action = risk_level(score)
    label = normalize_label(label)
    color = risk_color(score)

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### Score de risque")
        st.markdown(
            f"""
            <div style="display:flex;justify-content:space-between;gap:18px;align-items:flex-start;flex-wrap:wrap;">
                <div class="risk-score">{score * 100:.0f}<span>/100</span></div>
                <div class="risk-badge" style="background:{color};">Risque {label}</div>
            </div>
            <div class="gauge-shell">
                <div class="gauge-fill" style="width:{min(max(score, 0), 1) * 100:.1f}%"></div>
            </div>
            <div style="display:flex;justify-content:space-between;color:#6B7280;font-size:0.78rem;font-weight:750;">
                <span>Faible</span><span>Modere</span><span>Eleve</span>
            </div>
            <div class="action-box">
                <div class="action-title">Action recommandee</div>
                <div class="action-text">{action}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    lower_left, lower_right = st.columns([0.52, 0.48], gap="large")
    with lower_left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### Explicabilite")
        st.caption("Importance indicative des signaux pour le dossier courant. Ces elements orientent l'analyse agent.")
        render_explainability(model_importance(model, columns, case))
        st.markdown("</div>", unsafe_allow_html=True)

    with lower_right:
        st.markdown(
            """
            <div class="human-card">
                <h3>Decision humaine obligatoire</h3>
                <p>
                    FrauDetectAI ne sanctionne pas automatiquement. Le score classe les dossiers
                    par priorite de controle, puis un agent public verifie les pieces, contextualise
                    l'alerte et prend la decision finale selon les procedures administratives.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.expander("Voir les variables transmises au modele"):
            st.dataframe(case.T.rename(columns={0: "valeur"}), use_container_width=True, height=360)

elif page == "Modele ML":
    st.markdown('<div class="section-title">Performance du modele</div>', unsafe_allow_html=True)
    col_a, col_b = st.columns([0.48, 0.52], gap="large")
    with col_a:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### Modele selectionne")
        st.write(f"**{report.name}**")
        st.write(
            "Le seuil operationnel est calibre pour conserver un taux de faux positifs faible, "
            "tout en maintenant une capacite de detection utile pour les agents."
        )
        st.markdown("</div>", unsafe_allow_html=True)
    with col_b:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### Metriques")
        metrics = pd.DataFrame(
            [
                {"KPI": "AUC-PR", "Valeur": report.average_precision},
                {"KPI": "ROC-AUC", "Valeur": report.roc_auc},
                {"KPI": "Precision", "Valeur": report.precision},
                {"KPI": "Recall", "Valeur": report.recall},
                {"KPI": "F1-score", "Valeur": report.f1},
                {"KPI": "False positive rate", "Valeur": report.false_positive_rate},
            ]
        )
        st.dataframe(metrics, hide_index=True, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

elif page == "Gouvernance":
    st.markdown('<div class="section-title">Gouvernance IA responsable</div>', unsafe_allow_html=True)
    gov_cols = st.columns(3, gap="large")
    cards = [
        (
            "Supervision humaine",
            "Aucune decision automatique ayant un effet juridique. L'agent garde la responsabilite finale.",
        ),
        (
            "Auditabilite",
            "Chaque score doit etre rattache a une version de modele, une date, des donnees d'entree et un journal d'acces.",
        ),
        (
            "Monitoring",
            "Suivi mensuel de l'AUC-PR, du recall, des faux positifs, du drift des scores et des biais potentiels.",
        ),
    ]
    for column, (title, text) in zip(gov_cols, cards):
        with column:
            st.markdown(
                f"""
                <div class="card">
                    <h4>{title}</h4>
                    <p style="color:#6B7280;line-height:1.65;margin-bottom:0;">{text}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown('<div class="card" style="margin-top:18px;">', unsafe_allow_html=True)
    st.markdown("#### Cadre RGPD et AI Act")
    st.write(
        "Le systeme doit etre documente comme une IA a haut risque: base legale, minimisation des donnees, "
        "duree de conservation, droit de recours, explicabilite et revue humaine des dossiers sensibles."
    )
    st.markdown("</div>", unsafe_allow_html=True)
