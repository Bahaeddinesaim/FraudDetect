from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.config import PALETTE


FEATURE_LABELS = {
    "amount": "Montant",
    "hour": "Heure",
    "duplicate_count": "Doublons",
    "doc_mismatch": "Incoherence documentaire",
    "velocity": "Velocite",
    "prior_incidents": "Antecedents",
    "processing_hours": "Temps traitement",
}


def contribution_table(case: pd.Series) -> pd.DataFrame:
    values = {
        "amount": min(float(case["amount"]) / 25000, 1) * 35,
        "hour": (20 if int(case["hour"]) <= 5 or int(case["hour"]) >= 22 else -5),
        "duplicate_count": min(int(case["duplicate_count"]) * 6, 28),
        "doc_mismatch": float(case["doc_mismatch"]) * 32,
        "velocity": min(int(case["velocity"]) * 3, 24),
        "prior_incidents": min(int(case["prior_incidents"]) * 10, 30),
        "processing_hours": min(float(case["processing_hours"]) * 1.5, 18),
    }
    rows = [{"facteur": FEATURE_LABELS[k], "contribution": round(v, 2)} for k, v in values.items()]
    return pd.DataFrame(rows).sort_values("contribution", ascending=False)


def shap_contribution_table(cases: pd.DataFrame, case: pd.Series) -> tuple[pd.DataFrame, str]:
    """Return SHAP contributions when optional dependencies are available.

    The production Random Forest is preserved in ``models/``. For the synthetic
    GovTech portfolio, we train a small surrogate tree model on the same visible
    risk features so SHAP can explain the displayed case consistently. If SHAP
    or sklearn is unavailable, the deterministic business contribution fallback
    keeps the page operational.
    """
    try:
        import shap
        from sklearn.ensemble import RandomForestRegressor

        feature_cols = ["amount", "hour", "duplicate_count", "doc_mismatch", "velocity", "prior_incidents", "processing_hours"]
        train = cases[feature_cols].astype(float)
        target = cases["risk_score"].astype(float)
        surrogate = RandomForestRegressor(n_estimators=120, max_depth=6, random_state=42)
        surrogate.fit(train, target)
        explainer = shap.TreeExplainer(surrogate)
        one = case[feature_cols].astype(float).to_frame().T
        values = explainer.shap_values(one)
        values = np.asarray(values).reshape(-1)
        df = pd.DataFrame(
            {
                "facteur": [FEATURE_LABELS[c] for c in feature_cols],
                "contribution": np.round(values, 2),
            }
        ).sort_values("contribution", key=lambda s: s.abs(), ascending=False)
        return df, "SHAP TreeExplainer sur modele surrogate Random Forest"
    except Exception:
        return contribution_table(case), "Fallback explicabilite metier"


def importance_figure(cases: pd.DataFrame) -> go.Figure:
    rows = []
    for _, case in cases.head(200).iterrows():
        rows.append(contribution_table(case))
    df = pd.concat(rows).groupby("facteur", as_index=False)["contribution"].mean().sort_values("contribution")
    fig = go.Figure(go.Bar(x=df["contribution"], y=df["facteur"], orientation="h", marker_color=PALETTE["accent"]))
    fig.update_layout(title="Importance moyenne des variables", height=360)
    return fig


def waterfall_figure(case: pd.Series, contrib: pd.DataFrame | None = None) -> go.Figure:
    if contrib is None:
        contrib = contribution_table(case)
    fig = go.Figure(
        go.Waterfall(
            name="Risque",
            orientation="v",
            measure=["relative"] * len(contrib) + ["total"],
            x=list(contrib["facteur"]) + ["Score"],
            y=list(contrib["contribution"]) + [0],
            connector={"line": {"color": "#CBD5E1"}},
            increasing={"marker": {"color": PALETTE["danger"]}},
            decreasing={"marker": {"color": PALETTE["success"]}},
            totals={"marker": {"color": PALETTE["primary"]}},
        )
    )
    fig.update_layout(title="Waterfall des contributions au risque", height=420)
    return fig
