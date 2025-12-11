"""
ML API for Company Profit/Loss Prediction
==========================================
FastAPI application that serves an XGBoost model trained to predict
company profit/loss (GANANCIA/PÉRDIDA) based on financial and macroeconomic data.

Developed for the SuperSociedades 2025 Open Data Challenge.
"""

import os
from contextlib import asynccontextmanager
from typing import List

import numpy as np
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from schemas import (
    CompanyFinancialData,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    ModelInfoResponse,
    ErrorResponse,
    RegionEnum,
    DepartamentoEnum,
    MacrosectorEnum,
    CIIUEnum,
)

# Model configuration
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "xgboost_model.json")
API_VERSION = "1.0.0"

# Global model variable
model = None

# Feature columns in the exact order expected by the model
FEATURE_COLUMNS = [
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
    # Region one-hot encoded
    "REGIÓN_Antioquia",
    "REGIÓN_Bogotá - Cundinamarca",
    "REGIÓN_Centro - Oriente",
    "REGIÓN_Costa Atlántica",
    "REGIÓN_Costa Pacífica",
    "REGIÓN_Eje Cafetero",
    "REGIÓN_Otros",
    # Departamento one-hot encoded
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
    # Macrosector one-hot encoded
    "MACROSECTOR_AGROPECUARIO",
    "MACROSECTOR_COMERCIO",
    "MACROSECTOR_CONSTRUCCIÓN",
    "MACROSECTOR_MANUFACTURA",
    "MACROSECTOR_MINERO",
    "MACROSECTOR_SERVICIOS",
    # CIIU one-hot encoded
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

# Top 10 most important features (from training)
TOP_FEATURES = {
    "INGRESOS OPERACIONALES": 0.3599,
    "TOTAL PATRIMONIO": 0.1004,
    "MACROSECTOR_COMERCIO": 0.0813,
    "REGIÓN_Costa Pacífica": 0.0656,
    "Year": 0.0649,
    "MACROSECTOR_MINERO": 0.0513,
    "REGIÓN_Costa Atlántica": 0.0420,
    "TOTAL ACTIVOS": 0.0359,
    "MACROSECTOR_SERVICIOS": 0.0308,
    "MACROSECTOR_MANUFACTURA": 0.0298,
}


def load_model():
    """Load the XGBoost model from file."""
    global model
    if os.path.exists(MODEL_PATH):
        # Load as Booster for compatibility with different XGBoost versions
        model = xgb.Booster()
        model.load_model(MODEL_PATH)
        return True
    return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler - loads model on startup."""
    success = load_model()
    if not success:
        print(f"WARNING: Model file not found at {MODEL_PATH}")
    else:
        print(f"Model loaded successfully from {MODEL_PATH}")
    yield
    # Cleanup (if needed)
    print("Shutting down...")


# Create FastAPI app with OpenAPI documentation
app = FastAPI(
    title="Company Profit/Loss Prediction API",
    description="""
## Overview

This API provides predictions for company profit/loss (GANANCIA/PÉRDIDA) using an XGBoost machine learning model 
trained on Colombian company data from the SuperSociedades dataset.

## Usage

All financial values should be **normalized (0-1 scale)** before sending to the API.
The model expects the same preprocessing applied during training.

## Developed for

SuperSociedades 2025 Open Data Challenge (Reto Datos Abiertos)
    """,
    version=API_VERSION,
    contact={
        "name": "SuperSociedades Challenge Team",
        "url": "https://github.com/SantiagoMaya10/front-end-datos-abiertos-2025",
    },
    license_info={
        "name": "MIT License",
    },
    openapi_tags=[
        {
            "name": "predictions",
            "description": "Endpoints for making profit/loss predictions",
        },
        {
            "name": "model",
            "description": "Endpoints for model information and metadata",
        },
        {
            "name": "health",
            "description": "Health check and status endpoints",
        },
    ],
    lifespan=lifespan,
)

# Configure CORS for web access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def prepare_features(data: CompanyFinancialData) -> pd.DataFrame:
    """
    Transform input data into the feature format expected by the model.
    Performs one-hot encoding for categorical variables.
    """
    # Initialize all features with zeros
    features = {col: 0.0 for col in FEATURE_COLUMNS}
    
    # Set numerical features
    features["INGRESOS OPERACIONALES"] = data.ingresos_operacionales
    features["TOTAL ACTIVOS"] = data.total_activos
    features["TOTAL PASIVOS"] = data.total_pasivos
    features["TOTAL PATRIMONIO"] = data.total_patrimonio
    features["Year"] = float(data.year)
    features["TRM_Promedio"] = data.trm_promedio
    features["IPC_Promedio"] = data.ipc_promedio if data.ipc_promedio is not None else np.nan
    features["Inflacion_Promedio"] = data.inflacion_promedio
    features["Inversion_Extranjera_Total"] = data.inversion_extranjera_total
    features["PIB_Departamental"] = data.pib_departamental
    
    # One-hot encode Region
    region_col = f"REGIÓN_{data.region.value}"
    if region_col in features:
        features[region_col] = 1.0
    
    # One-hot encode Departamento
    dept_col = f"Departamento_{data.departamento.value}"
    if dept_col in features:
        features[dept_col] = 1.0
    
    # One-hot encode Macrosector
    macro_col = f"MACROSECTOR_{data.macrosector.value}"
    if macro_col in features:
        features[macro_col] = 1.0
    
    # One-hot encode CIIU
    ciiu_col = f"CIIU_{data.ciiu.value}"
    if ciiu_col in features:
        features[ciiu_col] = 1.0
    
    # Create DataFrame with correct column order
    df = pd.DataFrame([features])[FEATURE_COLUMNS]
    return df


def interpret_prediction(prediction: float) -> str:
    """Generate human-readable interpretation of the prediction."""
    if prediction > 1.0:
        return f"Strong positive profit expected (high value: {prediction:.2f})"
    elif prediction > 0.5:
        return "Positive profit expected (above average)"
    elif prediction > 0.0:
        return "Modest positive profit expected (below average)"
    elif prediction > -0.5:
        return "Slight loss expected"
    else:
        return "Significant loss expected"


# ============== API Endpoints ==============

@app.get("/", include_in_schema=False)
async def root():
    """Redirect root to API documentation."""
    return RedirectResponse(url="/docs")


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["health"],
    summary="Health Check",
    description="Check if the API is running and the model is loaded.",
)
async def health_check():
    """Return the health status of the API."""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        version=API_VERSION,
    )


@app.get(
    "/model/info",
    response_model=ModelInfoResponse,
    tags=["model"],
    summary="Get Model Information",
    description="Get detailed information about the ML model, including features and training details.",
)
async def get_model_info():
    """Return information about the loaded ML model."""
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded",
        )
    
    return ModelInfoResponse(
        model_type="XGBoost Regressor",
        target_variable="GANANCIA (PÉRDIDA) - Company Profit/Loss",
        num_features=len(FEATURE_COLUMNS),
        feature_names=FEATURE_COLUMNS,
        top_features=TOP_FEATURES,
        training_info={
            "dataset_size": 40000,
            "train_test_split": "80/20",
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 6,
            "objective": "reg:squarederror",
            "rmse": 0.2700,
            "mae": 0.0150,
            "r2_score": 0.1465,
        },
    )


@app.get(
    "/model/features",
    tags=["model"],
    summary="List Available Feature Options",
    description="Get the available options for categorical features (regions, departments, macrosectors, CIIU codes).",
)
async def get_feature_options():
    """Return available options for categorical features."""
    return {
        "regions": [r.value for r in RegionEnum],
        "departamentos": [d.value for d in DepartamentoEnum],
        "macrosectors": [m.value for m in MacrosectorEnum],
        "ciiu_codes": [c.value for c in CIIUEnum],
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["predictions"],
    summary="Predict Profit/Loss for a Company",
    description="""
Make a prediction for a single company based on its financial data.

**Important Notes:**
- All numerical financial values must be normalized (0-1 scale)
- The prediction is also in normalized scale
- Higher values indicate higher expected profit
    """,
    responses={
        200: {"description": "Successful prediction"},
        503: {"description": "Model not loaded", "model": ErrorResponse},
    },
)
async def predict_single(data: CompanyFinancialData):
    """Make a profit/loss prediction for a single company."""
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please check if the model file exists.",
        )
    
    try:
        # Prepare features
        features_df = prepare_features(data)
        
        # Make prediction using DMatrix for Booster
        dmatrix = xgb.DMatrix(features_df)
        prediction = model.predict(dmatrix)[0]
        
        return PredictionResponse(
            predicted_profit_loss=float(prediction),
            prediction_interpretation=interpret_prediction(float(prediction)),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction error: {str(e)}",
        )


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    tags=["predictions"],
    summary="Batch Predict Profit/Loss for Multiple Companies",
    description="""
Make predictions for multiple companies in a single request.

**Limits:**
- Minimum: 1 company
- Maximum: 1000 companies per request
    """,
    responses={
        200: {"description": "Successful batch prediction"},
        503: {"description": "Model not loaded", "model": ErrorResponse},
    },
)
async def predict_batch(request: BatchPredictionRequest):
    """Make profit/loss predictions for multiple companies."""
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please check if the model file exists.",
        )
    
    try:
        predictions = []
        
        for company_data in request.companies:
            features_df = prepare_features(company_data)
            # Make prediction using DMatrix for Booster
            dmatrix = xgb.DMatrix(features_df)
            prediction = model.predict(dmatrix)[0]
            
            predictions.append(
                PredictionResponse(
                    predicted_profit_loss=float(prediction),
                    prediction_interpretation=interpret_prediction(float(prediction)),
                )
            )
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_companies=len(predictions),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction error: {str(e)}",
        )


# Run with: uvicorn main:app --reload
if __name__ == "__main__":
    import uvicorn
    
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
