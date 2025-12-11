"""Offline smoke test to verify the bundled model works."""

import json
from pathlib import Path

import pandas as pd

from app.main import FEATURE_ORDER, load_model


def main() -> None:
    payload = json.loads(Path("sample_payload.json").read_text())
    records = payload["records"]
    df = pd.DataFrame(records)[FEATURE_ORDER].astype(float)

    model = load_model()
    preds = model.predict(df)
    print("Predictions:", [float(p) for p in preds])


if __name__ == "__main__":
    main()
