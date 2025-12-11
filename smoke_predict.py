"""Offline smoke test to verify preprocessing + model inference."""

import json
from pathlib import Path

from app.main import FEATURE_ORDER, load_model, load_preprocessor


def main() -> None:
    payload = json.loads(Path("sample_payload.json").read_text())
    records = payload["records"]

    prep = load_preprocessor(FEATURE_ORDER)
    df = prep.transform(records)

    model = load_model()
    preds = model.predict(df)
    print("Predictions:", [float(p) for p in preds])


if __name__ == "__main__":
    main()
