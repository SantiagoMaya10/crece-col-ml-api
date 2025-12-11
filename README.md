# Supersociedades ML API

FastAPI service that exposes the trained XGBoost regression model stored in `models/xgboost_model.json`. Designed to deploy on Railway.

## Quick start (local)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- Health: http://localhost:8000/health
- Schema: http://localhost:8000/schema (ordered model vector)

## Input contract (raw fields)

Send **raw business fields**; the API reproduces the training preprocessing (MinMax scaling + one-hot encoding) before invoking the model. Required fields per record:

- Numeric: `INGRESOS_OPERACIONALES`, `TOTAL_ACTIVOS`, `TOTAL_PASIVOS`, `TOTAL_PATRIMONIO`, `Year`, `TRM_Promedio`, `IPC_Promedio` (nullable), `Inflacion_Promedio`, `Inversion_Extranjera_Total`, `PIB_Departamental`
- Categoricals: `REGION`, `Departamento`, `MACROSECTOR`, `CIIU` (raw code as string, e.g. "4,661")

Under the hood the server fits encoders/scalers on `Modelo-machine-learning/Dataset para visualización.csv` and aligns to the model's `feature_names`. Unknown CIIU values are mapped to `Other` (as in training top-20). Department strings are normalized to the cleaned labels used during training.

## Request/response

- Endpoint: `POST /predict`
- Body: `{ "records": [ {<raw fields>} ] }`
- Example payload is provided in `sample_payload.json` (raw values from the visualization dataset).

Example curl:
```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d @sample_payload.json
```

## Deploy to Railway

1. Push this repo to Railway.
2. Use the `web` service; Railway detects `Procfile` and installs `requirements.txt` (includes scikit-learn required by XGBoost's sklearn API).
3. Ensure Python 3.10+.
4. The app loads the model from `ml-api-in-python/models/xgboost_model.json` and the preprocessing fit data from `Modelo-machine-learning/Dataset para visualización.csv` (kept in the repo), so no extra environment variables are required.

## Files

- `app/main.py` — FastAPI app, preprocessing, predict endpoint.
- `models/xgboost_model.json` — exported XGBoost model from training.
- `openapi.yaml` — OpenAPI 3.0 spec to import into Postman / SwaggerHub.
- `sample_payload.json` — raw example payload for `/predict`.
- `smoke_predict.py` — offline check to run preprocessing + predictions locally.
