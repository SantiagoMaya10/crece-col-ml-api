from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator

# Ordered list of features used during training
FEATURE_ORDER: List[str] = [
    "INGRESOS OPERACIONALES",
    "TOTAL ACTIVOS",
    "TOTAL PASIVOS",
    "TOTAL PATRIMONIO",
    "Year",
    "TRM_Promedio",
    "IPC_Promedio",
    "Inflacion_Promedio",
    "Inversion_Extranjera_Total",
    "PIB_Departamental",
    "REGIÓN_Antioquia",
    "REGIÓN_Bogotá - Cundinamarca",
    "REGIÓN_Centro - Oriente",
    "REGIÓN_Costa Atlántica",
    "REGIÓN_Costa Pacífica",
    "REGIÓN_Eje Cafetero",
    "REGIÓN_Otros",
    "Departamento_AMAZONAS",
    "Departamento_ANTIOQUIA",
    "Departamento_ARAUCA",
    "Departamento_ARMENIA",
    "Departamento_ATLANTICO",
    "Departamento_BOGOTA",
    "Departamento_BOLIVAR",
    "Departamento_BOYACA",
    "Departamento_CALDAS",
    "Departamento_CAQUETA",
    "Departamento_CASANARE",
    "Departamento_CAUCA",
    "Departamento_CESAR",
    "Departamento_CHOCO",
    "Departamento_CORDOBA",
    "Departamento_CUNDINAMARCA",
    "Departamento_GUAINIA",
    "Departamento_GUAJIRA",
    "Departamento_GUAVIARE",
    "Departamento_HUILA",
    "Departamento_LA GUAJIRA",
    "Departamento_MAGDALENA",
    "Departamento_META",
    "Departamento_MONTERIA",
    "Departamento_NARINO",
    "Departamento_NORTE DE SANTANDER",
    "Departamento_PUTUMAYO",
    "Departamento_QUINDIO",
    "Departamento_RISARALDA",
    "Departamento_SAN ANDRES Y PROVIDENCIA",
    "Departamento_SANTANDER",
    "Departamento_SUCRE",
    "Departamento_TOLIMA",
    "Departamento_VALLE",
    "Departamento_VALLEDUPAR",
    "Departamento_VAUPES",
    "Departamento_VICHADA",
    "MACROSECTOR_AGROPECUARIO",
    "MACROSECTOR_COMERCIO",
    "MACROSECTOR_CONSTRUCCIÓN",
    "MACROSECTOR_MANUFACTURA",
    "MACROSECTOR_MINERO",
    "MACROSECTOR_SERVICIOS",
    "CIIU_1,410",
    "CIIU_2,229",
    "CIIU_4,111",
    "CIIU_4,290",
    "CIIU_4,511",
    "CIIU_4,530",
    "CIIU_4,620",
    "CIIU_4,631",
    "CIIU_4,645",
    "CIIU_4,659",
    "CIIU_4,663",
    "CIIU_4,664",
    "CIIU_4,690",
    "CIIU_4,711",
    "CIIU_4,731",
    "CIIU_6,810",
    "CIIU_7,112",
    "CIIU_8,010",
    "CIIU_8,610",
    "CIIU_8,621",
    "CIIU_Other",
]


class PredictionRecord(BaseModel):
    """Single observation with all required features."""

    __root__: Dict[str, float]

    @validator("__root__")
    def validate_features(cls, value: Dict[str, float]) -> Dict[str, float]:
        missing = [feat for feat in FEATURE_ORDER if feat not in value]
        if missing:
            raise ValueError(f"Missing features: {missing}")

        extra = [key for key in value.keys() if key not in FEATURE_ORDER]
        if extra:
            raise ValueError(f"Unexpected features: {extra}")

        return value


class PredictionRequest(BaseModel):
    records: List[PredictionRecord] = Field(..., description="List of records to score")

    @validator("records")
    def non_empty(cls, value: List[PredictionRecord]) -> List[PredictionRecord]:
        if not value:
            raise ValueError("records cannot be empty")
        return value


class PredictionResponse(BaseModel):
    predictions: List[float]
    model_version: str
    feature_order: List[str]


app = FastAPI(
    title="Supersociedades ML API",
    description=(
        "REST API for the trained XGBoost regression model predicting "
        "`GANANCIA (PÉRDIDA)` based on 81 engineered features."
    ),
    version="1.0.0",
)

model: xgb.XGBRegressor | None = None


def load_model() -> xgb.XGBRegressor:
    """Load the serialized XGBoost model from disk."""
    model_path = Path(__file__).resolve().parent.parent / "models" / "xgboost_model.json"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    reg = xgb.XGBRegressor()
    reg.load_model(model_path)
    return reg


@app.on_event("startup")
def startup_event() -> None:
    global model
    model = load_model()


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest) -> PredictionResponse:
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        # Extract list of dicts from pydantic models
        records = [record.__root__ for record in payload.records]
        df = pd.DataFrame(records)
        # Ensure consistent column order
        df = df[FEATURE_ORDER]
        # Enforce numeric dtype
        df = df.astype(float)
    except Exception as exc:  # broad to surface validation issues
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        preds = model.predict(df)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Model inference failed: {exc}") from exc

    model_path = Path(__file__).resolve().parent.parent / "models" / "xgboost_model.json"
    model_version = model_path.stat().st_mtime_ns

    return PredictionResponse(
        predictions=[float(p) for p in preds],
        model_version=str(model_version),
        feature_order=FEATURE_ORDER,
    )


@app.get("/")
def root() -> Dict[str, str]:
    return {"message": "Supersociedades ML API. Visit /docs for Swagger UI."}
