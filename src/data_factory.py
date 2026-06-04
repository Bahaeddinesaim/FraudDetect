from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import streamlit as st

from src.config import REGION_COORDS


def classify_risk(score: float) -> str:
    if score >= 85:
        return "Critique"
    if score >= 68:
        return "Eleve"
    if score >= 42:
        return "Moyen"
    return "Faible"


@st.cache_data(show_spinner=False)
def make_fraud_cases(n: int = 780) -> pd.DataFrame:
    rng = np.random.default_rng(2026)
    today = date.today()
    regions = list(REGION_COORDS)
    fraud_types = [
        "Identite usurpee",
        "Double demande",
        "Faux justificatif",
        "Coordonnees bancaires suspectes",
        "Montant atypique",
        "Usager fictif",
        "Depot massif",
    ]
    user_profiles = ["Citoyen", "Entreprise", "Association", "Agent interne", "Mandataire"]
    services = ["Aides sociales", "Subventions", "Fiscalite", "Sante", "Logement", "Formation"]
    channels = ["Portail web", "Mobile", "Guichet", "API partenaire", "Courrier"]
    statuses = ["Nouveau", "En investigation", "Valide humain", "Rejete", "Clos"]

    rows = []
    for idx in range(n):
        region = rng.choice(regions, p=[0.22, 0.13, 0.10, 0.10, 0.10, 0.08, 0.10, 0.06, 0.05, 0.06])
        lat, lon = REGION_COORDS[region]
        service = rng.choice(services)
        fraud_type = rng.choice(fraud_types, p=[0.16, 0.14, 0.18, 0.13, 0.19, 0.10, 0.10])
        profile = rng.choice(user_profiles, p=[0.56, 0.18, 0.10, 0.04, 0.12])
        channel = rng.choice(channels, p=[0.46, 0.18, 0.13, 0.14, 0.09])
        submitted = today - timedelta(days=int(rng.integers(0, 180)))
        hour = int(np.clip(rng.normal(13, 5), 0, 23))
        if rng.random() < 0.16:
            hour = int(rng.choice([0, 1, 2, 3, 22, 23]))
        amount = float(rng.lognormal(mean=8.1, sigma=1.15))
        duplicate_count = int(rng.poisson(0.6) + (fraud_type in ["Double demande", "Depot massif"]) * rng.integers(2, 8))
        doc_mismatch = float(np.clip(rng.beta(1.4, 4.0) + 0.32 * (fraud_type in ["Faux justificatif", "Identite usurpee"]), 0, 1))
        velocity = int(rng.poisson(2) + (fraud_type == "Depot massif") * rng.integers(4, 15))
        prior_incidents = int(rng.poisson(0.3) + (profile in ["Mandataire", "Agent interne"]) * rng.integers(0, 3))
        processing_hours = float(np.clip(rng.normal(4.6, 1.9) + 2.4 * (amount > 12000), 0.4, 18))
        base = (
            18
            + min(amount / 900, 28)
            + duplicate_count * 4.8
            + doc_mismatch * 28
            + velocity * 1.8
            + prior_incidents * 8
            + (hour <= 5 or hour >= 22) * 10
            + (profile == "Agent interne") * 14
            + rng.normal(0, 8)
        )
        score = int(np.clip(base, 3, 99))
        is_fraud = score >= 68 or rng.random() < max(0.02, score / 650)
        risk_level = classify_risk(score)
        lat_j = lat + float(rng.normal(0, 0.12))
        lon_j = lon + float(rng.normal(0, 0.12))
        rows.append(
            {
                "case_id": f"FD-{today.year}-{idx + 10001}",
                "submitted_at": pd.Timestamp(submitted) + pd.Timedelta(hours=hour),
                "week": pd.Timestamp(submitted).to_period("W").start_time,
                "day_name": pd.Timestamp(submitted).day_name(),
                "hour": hour,
                "region": region,
                "city": region.split("-")[0],
                "lat": lat_j,
                "lon": lon_j,
                "service": service,
                "fraud_type": fraud_type,
                "user_profile": profile,
                "channel": channel,
                "amount": round(amount, 2),
                "risk_score": score,
                "risk_level": risk_level,
                "is_fraud": bool(is_fraud),
                "status": rng.choice(statuses, p=[0.28, 0.32, 0.14, 0.12, 0.14]),
                "processing_hours": round(processing_hours, 2),
                "duplicate_count": duplicate_count,
                "doc_mismatch": round(doc_mismatch, 3),
                "velocity": velocity,
                "prior_incidents": prior_incidents,
                "model_confidence": round(float(np.clip(0.54 + score / 220 + rng.normal(0, 0.05), 0.50, 0.99)), 3),
            }
        )
    return pd.DataFrame(rows).sort_values("submitted_at", ascending=False).reset_index(drop=True)


def filter_cases(
    cases: pd.DataFrame,
    regions: list[str],
    risk_levels: list[str],
    statuses: list[str],
    date_range: tuple[date, date] | list[date],
) -> pd.DataFrame:
    if not len(cases):
        return cases
    start, end = date_range
    mask = (
        cases["region"].isin(regions)
        & cases["risk_level"].isin(risk_levels)
        & cases["status"].isin(statuses)
        & (cases["submitted_at"].dt.date >= start)
        & (cases["submitted_at"].dt.date <= end)
    )
    return cases.loc[mask].copy()


def case_feature_frame(cases: pd.DataFrame) -> pd.DataFrame:
    cols = ["amount", "hour", "duplicate_count", "doc_mismatch", "velocity", "prior_incidents", "processing_hours"]
    return cases[cols].astype(float).copy()
