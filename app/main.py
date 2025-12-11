from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "xgboost_model.json"
RAW_DATA_PATH = PROJECT_ROOT.parent / "Modelo-machine-learning" / "Dataset para visualización.csv"


def load_feature_order() -> List[str]:
    """
    Source of truth for the expected vector order comes from the model file.
    This prevents drift between code and the trained artifact.
    """
    obj = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    names = obj.get("learner", {}).get("feature_names")
    if not names:
        raise RuntimeError("feature_names not found inside model file")
    return names


FEATURE_ORDER: List[str] = load_feature_order()


class RawRecord(BaseModel):
    """
    Incoming raw features before one-hot encoding and scaling.
    These match the columns in `Dataset para visualización.csv`.
    """

    INGRESOS_OPERACIONALES: float
    TOTAL_ACTIVOS: float
    TOTAL_PASIVOS: float
    TOTAL_PATRIMONIO: float
    Year: int
    TRM_Promedio: float
    IPC_Promedio: Optional[float] = Field(None, description="Can be null; will be imputed with training median")
    Inflacion_Promedio: float
    Inversion_Extranjera_Total: float
    PIB_Departamental: float
    REGION: str = Field(..., description="E.g. Bogotá - Cundinamarca")
    Departamento: str = Field(..., description="E.g. BOGOTA")
    MACROSECTOR: str = Field(..., description="E.g. COMERCIO, MANUFACTURA")
    CIIU: str = Field(..., description="Raw CIIU code as string, e.g. '4,661'")

    @validator("REGION", "Departamento", "MACROSECTOR", "CIIU", pre=True)
    def strip_strings(cls, v: str) -> str:
        return v.strip() if isinstance(v, str) else v


class PredictionRequest(BaseModel):
    records: List[RawRecord] = Field(..., description="List of records to score using raw business fields")

    @validator("records")
    def non_empty(cls, value: List[RawRecord]) -> List[RawRecord]:
        if not value:
            raise ValueError("records cannot be empty")
        return value


class PredictionResponse(BaseModel):
    predictions: List[float]
    model_version: str
    feature_order: List[str]


class Preprocessor:
    """
    Rebuild the minimal preprocessing used for training:
    - MinMax scale numeric columns (except Year)
    - One-hot encode REGION/Departamento/MACROSECTOR/CIIU (CIIU top-20 else Other)
    Feature order is taken from the model file to avoid drift.
    """

    numeric_to_scale = [
        "INGRESOS OPERACIONALES",
        "TOTAL ACTIVOS",
        "TOTAL PASIVOS",
        "TOTAL PATRIMONIO",
        "TRM_Promedio",
        "IPC_Promedio",
        "Inflacion_Promedio",
        "Inversion_Extranjera_Total",
        "PIB_Departamental",
    ]
    year_col = "Year"
    cat_cols = ["REGIÓN", "Departamento", "MACROSECTOR", "CIIU"]

    def __init__(self, feature_order: List[str]) -> None:
        self.feature_order = feature_order
        self.scaler = MinMaxScaler()
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.top_ciiu: set[str] = set()
        self.cat_feature_names: List[str] = []
        self.numeric_medians: Dict[str, float] = {}

    @staticmethod
    def _normalize_department(value: str) -> str:
        mapping = {
            "BOGOTÁ D.C.": "BOGOTA",
            "BOGOTA D.C.": "BOGOTA",
            "VALLE DEL CAUCA": "VALLE",
            "ATLÁNTICO": "ATLANTICO",
            "BOLÍVAR": "BOLIVAR",
            "CÓRDOBA": "CORDOBA",
            "NARIÑO": "NARINO",
            "CHOCÓ": "CHOCO",
            "BOYACÁ": "BOYACA",
            "QUINDÍO": "QUINDIO",
            "CAQUETÁ": "CAQUETA",
            "GUAINÍA": "GUAINIA",
            "VAUPÉS": "VAUPES",
            "SAN ANDRÉS Y PROVIDENCIA": "SAN ANDRES Y PROVIDENCIA",
        }
        return mapping.get(value, value)

    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["Departamento"] = df["Departamento"].apply(self._normalize_department)
        # CIIU top-20 handling
        df["CIIU"] = df["CIIU"].astype(str)
        if not self.top_ciiu:
            self.top_ciiu = set(df["CIIU"].value_counts().nlargest(20).index)
        df["CIIU"] = df["CIIU"].where(df["CIIU"].isin(self.top_ciiu), "Other")
        return df

    def fit(self, df_raw: pd.DataFrame) -> None:
        df = self._prepare_dataframe(df_raw)
        for col in self.numeric_to_scale:
            self.numeric_medians[col] = float(df[col].median())
            df[col] = df[col].fillna(self.numeric_medians[col])

        self.scaler.fit(df[self.numeric_to_scale])
        self.encoder.fit(df[self.cat_cols])
        self.cat_feature_names = list(self.encoder.get_feature_names_out(self.cat_cols))

    def transform(self, records: List[RawRecord | Dict[str, object]]) -> pd.DataFrame:
        normalized_records: List[Dict[str, object]] = []
        for r in records:
            if isinstance(r, RawRecord):
                normalized_records.append(r.dict())
            else:
                normalized_records.append(r)

        df_input = pd.DataFrame(normalized_records)
        df_input = df_input.rename(
            columns={
                "INGRESOS_OPERACIONALES": "INGRESOS OPERACIONALES",
                "TOTAL_ACTIVOS": "TOTAL ACTIVOS",
                "TOTAL_PASIVOS": "TOTAL PASIVOS",
                "TOTAL_PATRIMONIO": "TOTAL PATRIMONIO",
                "REGION": "REGIÓN",
            }
        )
        df_input["Departamento"] = df_input["Departamento"].apply(self._normalize_department)
        df_input["CIIU"] = df_input["CIIU"].astype(str)
        df_input["CIIU"] = df_input["CIIU"].where(df_input["CIIU"].isin(self.top_ciiu), "Other")
        for col in self.numeric_to_scale:
            df_input[col] = df_input[col].fillna(self.numeric_medians.get(col, 0.0))

        scaled = self.scaler.transform(df_input[self.numeric_to_scale])
        df_scaled = pd.DataFrame(scaled, columns=self.numeric_to_scale)

        encoded = self.encoder.transform(df_input[self.cat_cols])
        df_encoded = pd.DataFrame(encoded, columns=self.cat_feature_names)

        df_year = df_input[[self.year_col]]

        assembled = pd.concat([df_scaled, df_year, df_encoded], axis=1)

        # Ensure all expected columns exist
        for col in self.feature_order:
            if col not in assembled.columns:
                assembled[col] = 0.0

        assembled = assembled[self.feature_order]
        return assembled


app = FastAPI(
    title="Supersociedades ML API",
    description=(
        "REST API for the trained XGBoost regression model predicting "
        "`GANANCIA (PÉRDIDA)` based on 81 engineered features. "
        "Raw business fields are accepted and transformed to the training vector order."
    ),
    version="1.1.0",
)

model: xgb.XGBRegressor | None = None
preprocessor: Preprocessor | None = None


def load_model() -> xgb.XGBRegressor:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    reg = xgb.XGBRegressor()
    reg.load_model(MODEL_PATH)
    return reg


def load_preprocessor(feature_order: List[str]) -> Preprocessor:
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"Training raw dataset not found: {RAW_DATA_PATH}")

    df_raw = pd.read_csv(RAW_DATA_PATH)
    required_cols = [
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
        "REGIÓN",
        "Departamento",
        "MACROSECTOR",
        "CIIU",
    ]
    missing = [c for c in required_cols if c not in df_raw.columns]
    if missing:
        raise RuntimeError(f"Raw dataset missing required columns: {missing}")

    prep = Preprocessor(feature_order)
    prep.fit(df_raw)
    return prep


@app.on_event("startup")
def startup_event() -> None:
    global model, preprocessor
    model = load_model()
    preprocessor = load_preprocessor(FEATURE_ORDER)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest) -> PredictionResponse:
    if model is None or preprocessor is None:
        raise HTTPException(status_code=500, detail="Model or preprocessor not loaded")

    try:
        df = preprocessor.transform(payload.records)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Preprocessing failed: {exc}") from exc

    try:
        preds = model.predict(df)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Model inference failed: {exc}") from exc

    model_version = MODEL_PATH.stat().st_mtime_ns

    return PredictionResponse(
        predictions=[float(p) for p in preds],
        model_version=str(model_version),
        feature_order=FEATURE_ORDER,
    )


@app.get("/")
def root() -> Dict[str, str]:
    return {"message": "Supersociedades ML API. Visit /docs for Swagger UI."}


@app.get("/schema")
def schema() -> Dict[str, List[str]]:
    """
    Return the expected feature order after preprocessing so clients can align payloads if needed.
    """
    return {"feature_order": FEATURE_ORDER}
