from __future__ import annotations

from pathlib import Path

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


st.set_page_config(page_title="FraudAI", page_icon="FAI", layout="wide")
st.title("FraudAI - Détection de fraude dans les services publics")

bundle = get_bundle()
model = bundle["model"]
report = bundle["report"]
columns = bundle["columns"]

metric_cols = st.columns(5)
metric_cols[0].metric("Modèle", report.name)
metric_cols[1].metric("AUC-PR", f"{report.average_precision:.3f}")
metric_cols[2].metric("Recall", f"{report.recall:.3f}")
metric_cols[3].metric("Precision", f"{report.precision:.3f}")
metric_cols[4].metric("Seuil", f"{report.threshold:.2f}")

st.divider()

left, right = st.columns([0.38, 0.62])
with left:
    st.subheader("Saisie agent")
    amount = st.number_input("Montant de la demande / transaction", min_value=0.0, value=128.0, step=10.0)
    hour = st.slider("Heure de dépôt", min_value=0, max_value=23, value=14)
    scenario = st.selectbox(
        "Profil de dossier",
        ["Standard", "Montant atypique", "Signaux incohérents", "Cas fortement suspect"],
    )
    submitted = st.button("Calculer le score de risque", type="primary")

def make_case() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    row = {feature: 0.0 for feature in FEATURES}
    row["Amount"] = amount
    row["Time"] = hour * 3600
    row.update({f"V{i}": rng.normal(0, 0.45) for i in range(1, 29)})
    if scenario == "Montant atypique":
        row["V1"], row["V7"], row["V14"] = 1.4, 1.3, -1.4
    elif scenario == "Signaux incohérents":
        row["V3"], row["V10"], row["V12"] = -1.6, -2.1, -1.7
    elif scenario == "Cas fortement suspect":
        row["V1"], row["V3"], row["V10"], row["V12"], row["V14"], row["V17"] = 2.2, -2.0, -2.7, -2.2, -3.0, -1.6
    df = pd.DataFrame([row])
    df["Hour"] = (df["Time"] // 3600) % 24
    df["LogAmount"] = np.log1p(df["Amount"])
    df["IsNight"] = df["Hour"].between(0, 5).astype(int)
    return df[columns]

with right:
    st.subheader("Score de risque")
    case = make_case()
    score = float(predict_proba_positive(model, case)[0])
    label, action = risk_level(score)
    st.progress(min(max(score, 0.0), 1.0))
    c1, c2, c3 = st.columns(3)
    c1.metric("Risque", f"{score * 100:.0f}/100")
    c2.metric("Niveau", label)
    c3.metric("Action", action)

    st.caption("La décision finale reste prise par un agent public. Le score sert à prioriser le contrôle, pas à sanctionner automatiquement.")
    st.dataframe(case.T.rename(columns={0: "valeur"}), use_container_width=True, height=430)

st.divider()
st.subheader("Gouvernance opérationnelle")
st.write(
    "Monitoring recommandé: AUC-PR, recall fraude, taux de faux positifs, dérive des montants, dérive des scores, "
    "revue humaine des dossiers à risque élevé et audit biais/RGPD tous les 6 mois."
)
