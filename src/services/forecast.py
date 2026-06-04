from __future__ import annotations

import numpy as np
import pandas as pd


def forecast_fraud(cases: pd.DataFrame, periods: int = 30) -> pd.DataFrame:
    daily = (
        cases.assign(day=cases["submitted_at"].dt.floor("D"))
        .groupby("day", as_index=False)
        .agg(fraudes=("is_fraud", "sum"), pertes=("amount", lambda s: s[cases.loc[s.index, "is_fraud"]].sum()), critiques=("risk_level", lambda s: (s == "Critique").sum()))
        .sort_values("day")
    )
    if len(daily) < 8:
        return pd.DataFrame()
    y = daily["fraudes"].astype(float).to_numpy()
    future_idx = pd.date_range(daily["day"].max() + pd.Timedelta(days=1), periods=periods, freq="D")
    try:
        from statsmodels.tsa.arima.model import ARIMA

        fitted = ARIMA(y, order=(1, 1, 1)).fit()
        pred = np.clip(np.asarray(fitted.forecast(periods), dtype=float), 0, None)
    except Exception:
        pred = None
    x = np.arange(len(y))
    if pred is None:
        trend = np.polyfit(x, y, 1)
        future_x = np.arange(len(y), len(y) + periods)
        rng = np.random.default_rng(7)
        pred = np.clip(np.polyval(trend, future_x) + rng.normal(0, max(y.std(), 1) * 0.35, periods), 0, None)
    loss_rate = max(float(daily["pertes"].mean()), 1)
    critical_rate = max(float(daily["critiques"].mean() / max(daily["fraudes"].mean(), 1)), 0.15)
    return pd.DataFrame(
        {
            "day": future_idx,
            "fraudes_prevues": pred.round(1),
            "pertes_potentielles": (pred * loss_rate).round(2),
            "cas_critiques_attendus": np.ceil(pred * critical_rate).astype(int),
        }
    )
