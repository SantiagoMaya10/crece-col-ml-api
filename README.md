# ML API - Company Profit/Loss Prediction

A FastAPI REST API for predicting company profit/loss (GANANCIA/PÉRDIDA) using an XGBoost machine learning model. Developed for the SuperSociedades 2025 Open Data Challenge.

## Features

- **Single Prediction**: Predict profit/loss for one company
- **Batch Prediction**: Predict for up to 1000 companies at once
- **OpenAPI Documentation**: Interactive Swagger UI and ReDoc
- **Health Checks**: Built-in health endpoint for Railway
- **CORS Enabled**: Ready for web application integration

## Quick Start

### Local Development

1. **Create a virtual environment**:
```bash
cd ml-api-in-python
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Run the API**:
```bash
uvicorn main:app --reload
```

4. **Access the API**:
   - API Documentation (Swagger): http://localhost:8000/docs
   - Alternative Docs (ReDoc): http://localhost:8000/redoc
   - OpenAPI Spec: http://localhost:8000/openapi.json

### Deploy to Railway

1. Push this folder to a GitHub repository
2. Connect your Railway project to the repository
3. Railway will auto-detect the Python project and deploy

The `railway.json` and `Procfile` are already configured for deployment.

## API Endpoints

### Health & Info

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/model/info` | Model information and metrics |
| GET | `/model/features` | Available categorical options |

### Predictions

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/predict` | Single company prediction |
| POST | `/predict/batch` | Batch predictions (up to 1000) |

## Example Request

### Single Prediction

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "ingresos_operacionales": 0.15,
    "total_activos": 0.12,
    "total_pasivos": 0.08,
    "total_patrimonio": 0.10,
    "year": 2022,
    "trm_promedio": 0.87,
    "ipc_promedio": null,
    "inflacion_promedio": 0.33,
    "inversion_extranjera_total": 0.0,
    "pib_departamental": 0.07,
    "region": "Bogotá - Cundinamarca",
    "departamento": "BOGOTA",
    "macrosector": "SERVICIOS",
    "ciiu": "Other"
  }'
```

### Response

```json
{
  "predicted_profit_loss": 0.45,
  "prediction_interpretation": "Modest positive profit expected (below average)"
}
```

## Input Data Format

All numerical financial values must be **normalized (0-1 scale)**:

| Field | Type | Description |
|-------|------|-------------|
| `ingresos_operacionales` | float (0-1) | Operational income (normalized) |
| `total_activos` | float (0-1) | Total assets (normalized) |
| `total_pasivos` | float (0-1) | Total liabilities (normalized) |
| `total_patrimonio` | float (0-1) | Total equity (normalized) |
| `year` | int | Fiscal year (2018-2030) |
| `trm_promedio` | float (0-1) | Average TRM exchange rate |
| `ipc_promedio` | float/null | Average CPI (can be null) |
| `inflacion_promedio` | float (0-1) | Average inflation rate |
| `inversion_extranjera_total` | float (0-1) | Foreign investment |
| `pib_departamental` | float (0-1) | Departmental GDP |
| `region` | string | Geographic region |
| `departamento` | string | Department code |
| `macrosector` | string | Economic sector |
| `ciiu` | string | CIIU activity code |

### Available Regions
- Antioquia
- Bogotá - Cundinamarca
- Centro - Oriente
- Costa Atlántica
- Costa Pacífica
- Eje Cafetero
- Otros

### Available Macrosectors
- AGROPECUARIO
- COMERCIO
- CONSTRUCCIÓN
- MANUFACTURA
- MINERO
- SERVICIOS

## Model Information

- **Algorithm**: XGBoost Regressor
- **Features**: 81 features (financial, macroeconomic, categorical)
- **Training Dataset**: 40,000 Colombian companies
- **R² Score**: 0.1465 (explains ~15% of variance)
- **MAE**: 0.015 (average error in normalized scale)
- **RMSE**: 0.27

### Top 10 Important Features

1. **INGRESOS OPERACIONALES** (36%) - Operational income
2. **TOTAL PATRIMONIO** (10%) - Total equity
3. **MACROSECTOR_COMERCIO** (8%) - Commerce sector
4. **REGIÓN_Costa Pacífica** (7%) - Pacific Coast region
5. **Year** (6%) - Temporal trends
6. **MACROSECTOR_MINERO** (5%) - Mining sector
7. **REGIÓN_Costa Atlántica** (4%) - Atlantic Coast region
8. **TOTAL ACTIVOS** (4%) - Total assets
9. **MACROSECTOR_SERVICIOS** (3%) - Services sector
10. **MACROSECTOR_MANUFACTURA** (3%) - Manufacturing sector

## Project Structure

```
ml-api-in-python/
├── main.py              # FastAPI application
├── schemas.py           # Pydantic models for validation
├── models/
│   └── xgboost_model.json  # Trained XGBoost model
├── requirements.txt     # Python dependencies
├── Procfile            # Railway/Heroku deployment
├── railway.json        # Railway configuration
└── README.md           # This file
```

## License

MIT License - Part of the SuperSociedades 2025 Open Data Challenge project.
