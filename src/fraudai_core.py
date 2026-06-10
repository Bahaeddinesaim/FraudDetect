from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler


RANDOM_STATE = 42
FEATURES = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]
TARGET = "Class"


@dataclass
class ModelReport:
    name: str
    threshold: float
    average_precision: float
    roc_auc: float
    precision: float
    recall: float
    f1: float
    false_positive_rate: float


def load_transactions(path: str | Path = "data/creditcard.csv", strict: bool = False) -> pd.DataFrame:
    path = Path(path)
    if path.exists():
        df = pd.read_csv(path)
        missing = [col for col in FEATURES + [TARGET] if col not in df.columns]
        if missing:
            raise ValueError(f"Colonnes manquantes dans {path}: {missing}")
        return df[FEATURES + [TARGET]].copy()
    if strict:
        raise FileNotFoundError(
            f"Base de donnees obligatoire introuvable: {path}. "
            "Place le fichier creditcard.csv fourni dans le dossier data/."
        )
    return make_synthetic_transactions()


def make_synthetic_transactions(n_rows: int = 12_000, fraud_rate: float = 0.012) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)
    y = rng.binomial(1, fraud_rate, size=n_rows)
    hard_legitimate = (y == 0) & (rng.random(n_rows) < 0.025)
    data: dict[str, Any] = {}
    data["Time"] = rng.integers(0, 172_800, size=n_rows)
    base_amount = rng.lognormal(mean=3.1, sigma=1.15, size=n_rows)
    data["Amount"] = np.where(y == 1, base_amount * rng.uniform(1.1, 3.0, n_rows), base_amount)
    data["Amount"] = np.where(hard_legitimate, data["Amount"] * rng.uniform(1.2, 2.5, n_rows), data["Amount"])

    signal_features = {1: 1.2, 3: -1.0, 7: 0.8, 10: -1.35, 12: -1.1, 14: -1.45, 17: -0.9}
    for idx in range(1, 29):
        shift = signal_features.get(idx, 0.0)
        noise = rng.normal(0, 1.25, size=n_rows)
        mimic = hard_legitimate * shift * rng.uniform(0.25, 0.7, n_rows)
        data[f"V{idx}"] = noise + y * shift + mimic + rng.normal(0, 0.35, size=n_rows)

    df = pd.DataFrame(data)
    df[TARGET] = y
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Hour"] = (out["Time"] // 3600) % 24
    out["LogAmount"] = np.log1p(out["Amount"].clip(lower=0))
    out["IsNight"] = out["Hour"].between(0, 5).astype(int)
    return out


def split_xy(df: pd.DataFrame):
    enriched = add_features(df)
    X = enriched.drop(columns=[TARGET])
    y = enriched[TARGET].astype(int)
    return X, y


def candidate_models() -> dict[str, Any]:
    return {
        "Logistic Regression + SMOTE": ImbPipeline(
            steps=[
                ("scaler", RobustScaler()),
                ("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=3)),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1500,
                        class_weight="balanced",
                        n_jobs=None,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=360,
            max_features="sqrt",
            min_samples_leaf=1,
            class_weight="balanced_subsample",
            oob_score=True,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=180,
            learning_rate=0.06,
            l2_regularization=0.02,
            random_state=RANDOM_STATE,
        ),
    }


def predict_proba_positive(model: Any, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    scores = model.decision_function(X)
    return 1 / (1 + np.exp(-scores))


def choose_threshold(y_true: pd.Series, scores: np.ndarray, max_fpr: float = 0.05) -> float:
    thresholds = np.linspace(0.02, 0.98, 97)
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in thresholds:
        pred = (scores >= threshold).astype(int)
        tn, fp, _, _ = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        fpr = fp / max(fp + tn, 1)
        if fpr > max_fpr:
            continue
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)
    return best_threshold


def evaluate_model(name: str, model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> ModelReport:
    scores = predict_proba_positive(model, X_test)
    threshold = choose_threshold(y_test, scores)
    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
    return ModelReport(
        name=name,
        threshold=threshold,
        average_precision=average_precision_score(y_test, scores),
        roc_auc=roc_auc_score(y_test, scores),
        precision=precision_score(y_test, pred, zero_division=0),
        recall=recall_score(y_test, pred, zero_division=0),
        f1=f1_score(y_test, pred, zero_division=0),
        false_positive_rate=fp / max(fp + tn, 1),
    )


def train_and_select(df: pd.DataFrame):
    X, y = split_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
    )
    reports: list[ModelReport] = []
    fitted: dict[str, Any] = {}
    for name, model in candidate_models().items():
        model.fit(X_train, y_train)
        fitted[name] = model
        reports.append(evaluate_model(name, model, X_test, y_test))
    best = max(reports, key=lambda report: (report.average_precision, report.f1))
    return fitted[best.name], best, reports, X_test, y_test


def risk_level(score: float) -> tuple[str, str]:
    risk = int(round(score * 100))
    if risk <= 30:
        return "Faible", "Traitement standard"
    if risk <= 70:
        return "Modere", "Controle selon capacite disponible"
    return "Eleve", "Controle prioritaire par un agent"


def save_bundle(path: str | Path, model: Any, report: ModelReport, columns: list[str]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "report": report, "columns": columns}, path)


def load_bundle(path: str | Path = "models/fraudai_model.joblib") -> dict[str, Any]:
    return joblib.load(path)
