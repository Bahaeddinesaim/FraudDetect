from __future__ import annotations

from pathlib import Path
from shutil import copy2

import kagglehub


DATASET = "mlg-ulb/creditcardfraud"
TARGET = Path("data/creditcard.csv")


def main() -> None:
    path = Path(kagglehub.dataset_download(DATASET))
    csv_files = list(path.rglob("creditcard.csv"))
    if not csv_files:
        raise FileNotFoundError(f"creditcard.csv introuvable dans {path}")

    TARGET.parent.mkdir(parents=True, exist_ok=True)
    copy2(csv_files[0], TARGET)

    print("Path to dataset files:", path)
    print("Dataset copied to:", TARGET.resolve())


if __name__ == "__main__":
    main()
