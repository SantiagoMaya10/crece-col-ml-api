"""
Pydantic schemas for the ML API request/response validation.
Based on the XGBoost model trained for predicting company profit/loss (GANANCIA/PÉRDIDA).
"""

from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum


class RegionEnum(str, Enum):
    """Available regions in Colombia"""
    ANTIOQUIA = "Antioquia"
    BOGOTA_CUNDINAMARCA = "Bogotá - Cundinamarca"
    CENTRO_ORIENTE = "Centro - Oriente"
    COSTA_ATLANTICA = "Costa Atlántica"
    COSTA_PACIFICA = "Costa Pacífica"
    EJE_CAFETERO = "Eje Cafetero"
    OTROS = "Otros"


class DepartamentoEnum(str, Enum):
    """Available departments in Colombia"""
    AMAZONAS = "AMAZONAS"
    ANTIOQUIA = "ANTIOQUIA"
    ARAUCA = "ARAUCA"
    ARMENIA = "ARMENIA"
    ATLANTICO = "ATLANTICO"
    BOGOTA = "BOGOTA"
    BOLIVAR = "BOLIVAR"
    BOYACA = "BOYACA"
    CALDAS = "CALDAS"
    CAQUETA = "CAQUETA"
    CASANARE = "CASANARE"
    CAUCA = "CAUCA"
    CESAR = "CESAR"
    CHOCO = "CHOCO"
    CORDOBA = "CORDOBA"
    CUNDINAMARCA = "CUNDINAMARCA"
    GUAINIA = "GUAINIA"
    GUAJIRA = "GUAJIRA"
    GUAVIARE = "GUAVIARE"
    HUILA = "HUILA"
    LA_GUAJIRA = "LA GUAJIRA"
    MAGDALENA = "MAGDALENA"
    META = "META"
    MONTERIA = "MONTERIA"
    NARINO = "NARINO"
    NORTE_DE_SANTANDER = "NORTE DE SANTANDER"
    PUTUMAYO = "PUTUMAYO"
    QUINDIO = "QUINDIO"
    RISARALDA = "RISARALDA"
    SAN_ANDRES_Y_PROVIDENCIA = "SAN ANDRES Y PROVIDENCIA"
    SANTANDER = "SANTANDER"
    SUCRE = "SUCRE"
    TOLIMA = "TOLIMA"
    VALLE = "VALLE"
    VALLEDUPAR = "VALLEDUPAR"
    VAUPES = "VAUPES"
    VICHADA = "VICHADA"


class MacrosectorEnum(str, Enum):
    """Available economic macrosectors"""
    AGROPECUARIO = "AGROPECUARIO"
    COMERCIO = "COMERCIO"
    CONSTRUCCION = "CONSTRUCCIÓN"
    MANUFACTURA = "MANUFACTURA"
    MINERO = "MINERO"
    SERVICIOS = "SERVICIOS"


class CIIUEnum(str, Enum):
    """Available CIIU (economic activity) codes"""
    CIIU_1410 = "1,410"
    CIIU_2229 = "2,229"
    CIIU_4111 = "4,111"
    CIIU_4290 = "4,290"
    CIIU_4511 = "4,511"
    CIIU_4530 = "4,530"
    CIIU_4620 = "4,620"
    CIIU_4631 = "4,631"
    CIIU_4645 = "4,645"
    CIIU_4659 = "4,659"
    CIIU_4663 = "4,663"
    CIIU_4664 = "4,664"
    CIIU_4690 = "4,690"
    CIIU_4711 = "4,711"
    CIIU_4731 = "4,731"
    CIIU_6810 = "6,810"
    CIIU_7112 = "7,112"
    CIIU_8010 = "8,010"
    CIIU_8610 = "8,610"
    CIIU_8621 = "8,621"
    CIIU_OTHER = "Other"


class CompanyFinancialData(BaseModel):
    """
    Financial data input for a company prediction.
    All numerical values should be normalized (0-1 scale).
    """
    # Core financial metrics (normalized 0-1)
    ingresos_operacionales: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Operational income (normalized 0-1). Main predictor ~36% importance.",
        json_schema_extra={"example": 0.15}
    )
    total_activos: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Total assets (normalized 0-1)",
        json_schema_extra={"example": 0.12}
    )
    total_pasivos: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Total liabilities (normalized 0-1)",
        json_schema_extra={"example": 0.08}
    )
    total_patrimonio: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Total equity/patrimony (normalized 0-1). ~10% importance.",
        json_schema_extra={"example": 0.10}
    )
    
    # Temporal feature
    year: int = Field(
        ...,
        ge=2018,
        le=2030,
        description="Fiscal year of the financial data",
        json_schema_extra={"example": 2022}
    )
    
    # Macroeconomic indicators (normalized 0-1)
    trm_promedio: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Average exchange rate TRM (normalized 0-1)",
        json_schema_extra={"example": 0.87}
    )
    ipc_promedio: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Average consumer price index IPC (normalized 0-1). Can be null.",
        json_schema_extra={"example": None}
    )
    inflacion_promedio: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Average inflation rate (normalized 0-1)",
        json_schema_extra={"example": 0.33}
    )
    inversion_extranjera_total: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Total foreign investment (normalized 0-1)",
        json_schema_extra={"example": 0.0}
    )
    pib_departamental: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Departmental GDP (normalized 0-1)",
        json_schema_extra={"example": 0.07}
    )
    
    # Categorical features
    region: RegionEnum = Field(
        ...,
        description="Geographic region of Colombia",
        json_schema_extra={"example": "Bogotá - Cundinamarca"}
    )
    departamento: DepartamentoEnum = Field(
        ...,
        description="Department (state) in Colombia",
        json_schema_extra={"example": "BOGOTA"}
    )
    macrosector: MacrosectorEnum = Field(
        ...,
        description="Economic macrosector of the company",
        json_schema_extra={"example": "SERVICIOS"}
    )
    ciiu: CIIUEnum = Field(
        ...,
        description="CIIU economic activity code",
        json_schema_extra={"example": "Other"}
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "ingresos_operacionales": 0.15,
                    "total_activos": 0.12,
                    "total_pasivos": 0.08,
                    "total_patrimonio": 0.10,
                    "year": 2022,
                    "trm_promedio": 0.87,
                    "ipc_promedio": None,
                    "inflacion_promedio": 0.33,
                    "inversion_extranjera_total": 0.0,
                    "pib_departamental": 0.07,
                    "region": "Bogotá - Cundinamarca",
                    "departamento": "BOGOTA",
                    "macrosector": "SERVICIOS",
                    "ciiu": "Other"
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    """Response model for a single prediction"""
    predicted_profit_loss: float = Field(
        ...,
        description="Predicted profit/loss value (normalized scale)",
        json_schema_extra={"example": 0.45}
    )
    prediction_interpretation: str = Field(
        ...,
        description="Human-readable interpretation of the prediction",
        json_schema_extra={"example": "Positive profit expected"}
    )


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions"""
    companies: List[CompanyFinancialData] = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="List of company financial data for batch prediction"
    )


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions"""
    predictions: List[PredictionResponse] = Field(
        ...,
        description="List of predictions corresponding to input companies"
    )
    total_companies: int = Field(
        ...,
        description="Total number of companies processed"
    )


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether the ML model is loaded")
    version: str = Field(..., description="API version")


class ModelInfoResponse(BaseModel):
    """Model information response"""
    model_type: str = Field(..., description="Type of ML model")
    target_variable: str = Field(..., description="Variable being predicted")
    num_features: int = Field(..., description="Number of input features")
    feature_names: List[str] = Field(..., description="List of feature names")
    top_features: dict = Field(..., description="Top 10 most important features with importance scores")
    training_info: dict = Field(..., description="Information about model training")


class ErrorResponse(BaseModel):
    """Error response model"""
    detail: str = Field(..., description="Error message")
    error_type: str = Field(..., description="Type of error")
