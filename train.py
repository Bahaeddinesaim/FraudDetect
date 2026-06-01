from pathlib import Path

import pandas as pd

from src.fraudai_core import load_transactions, save_bundle, split_xy, train_and_select


def main() -> None:
    df = load_transactions()
    model, best_report, reports, X_test, y_test = train_and_select(df)
    X, _ = split_xy(df)
    save_bundle("models/fraudai_model.joblib", model, best_report, list(X.columns))

    rows = [report.__dict__ for report in reports]
    metrics = pd.DataFrame(rows).sort_values("average_precision", ascending=False)
    Path("models").mkdir(exist_ok=True)
    metrics.to_csv("models/model_metrics.csv", index=False)

    print("FraudAI training complete")
    print(f"Rows: {len(df):,} | Fraud rate: {df['Class'].mean():.3%}")
    print(f"Best model: {best_report.name}")
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
