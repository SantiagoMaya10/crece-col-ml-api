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

## Request/response

- Endpoint: `POST /predict`
- Body shape: `{ "records": [ {<all 81 features>} ] }`
- All 81 training features are required for every record (see `feature_order` returned in the response or `openapi.yaml`).
- Example payload is provided in `sample_payload.json`.

Example curl:
```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d @sample_payload.json
```

## Deploy to Railway

1. Push this repo to Railway.
2. Use the `web` service; Railway detects `Procfile` and installs `requirements.txt`.
3. No extra environment variables are required (model file already bundled under `models/`).

## Files

- `app/main.py` — FastAPI app, model loading, predict endpoint.
- `models/xgboost_model.json` — exported XGBoost model from training.
- `openapi.yaml` — OpenAPI 3.0 spec to import into Postman / SwaggerHub.
- `sample_payload.json` — minimal example payload for `/predict`.
