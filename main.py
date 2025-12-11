import json
import os
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

# Initialize FastAPI app
app = FastAPI(
    title="Earnings Prediction API",
    description="API for predicting earnings (GANANCIA/PÉRDIDA) based on financial and regional data.",
    version="1.0.0"
)

# Load Model
MODEL_FILE = "xgboost_model.json"
model = None

def load_model():
    global model
    if os.path.exists(MODEL_FILE):
        # Use Booster directly to avoid Scikit-Learn wrapper version mismatches
        model = xgb.Booster()
        model.load_model(MODEL_FILE)
        print("Model loaded successfully.")
    else:
        print(f"Warning: {MODEL_FILE} not found. Predictions will fail.")

@app.on_event("startup")
async def startup_event():
    load_model()

# Input Schema
class PredictionInput(BaseModel):
    INGRESOS_OPERACIONALES: float = Field(..., alias='INGRESOS OPERACIONALES')
    TOTAL_ACTIVOS: float = Field(..., alias='TOTAL ACTIVOS')
    TOTAL_PASIVOS: float = Field(..., alias='TOTAL PASIVOS')
    TOTAL_PATRIMONIO: float = Field(..., alias='TOTAL PATRIMONIO')
    Year: float = Field(..., alias='Year')
    TRM_Promedio: float = Field(..., alias='TRM_Promedio')
    IPC_Promedio: float = Field(..., alias='IPC_Promedio')
    Inflacion_Promedio: float = Field(..., alias='Inflacion_Promedio')
    Inversion_Extranjera_Total: float = Field(..., alias='Inversion_Extranjera_Total')
    PIB_Departamental: float = Field(..., alias='PIB_Departamental')
    REGION_Antioquia: float = Field(..., alias='REGIÓN_Antioquia')
    REGION_Bogotá___Cundinamarca: float = Field(..., alias='REGIÓN_Bogotá - Cundinamarca')
    REGION_Centro___Oriente: float = Field(..., alias='REGIÓN_Centro - Oriente')
    REGION_Costa_Atlántica: float = Field(..., alias='REGIÓN_Costa Atlántica')
    REGION_Costa_Pacífica: float = Field(..., alias='REGIÓN_Costa Pacífica')
    REGION_Eje_Cafetero: float = Field(..., alias='REGIÓN_Eje Cafetero')
    REGION_Otros: float = Field(..., alias='REGIÓN_Otros')
    Departamento_AMAZONAS: float = Field(..., alias='Departamento_AMAZONAS')
    Departamento_ANTIOQUIA: float = Field(..., alias='Departamento_ANTIOQUIA')
    Departamento_ARAUCA: float = Field(..., alias='Departamento_ARAUCA')
    Departamento_ARMENIA: float = Field(..., alias='Departamento_ARMENIA')
    Departamento_ATLANTICO: float = Field(..., alias='Departamento_ATLANTICO')
    Departamento_BOGOTA: float = Field(..., alias='Departamento_BOGOTA')
    Departamento_BOLIVAR: float = Field(..., alias='Departamento_BOLIVAR')
    Departamento_BOYACA: float = Field(..., alias='Departamento_BOYACA')
    Departamento_CALDAS: float = Field(..., alias='Departamento_CALDAS')
    Departamento_CAQUETA: float = Field(..., alias='Departamento_CAQUETA')
    Departamento_CASANARE: float = Field(..., alias='Departamento_CASANARE')
    Departamento_CAUCA: float = Field(..., alias='Departamento_CAUCA')
    Departamento_CESAR: float = Field(..., alias='Departamento_CESAR')
    Departamento_CHOCO: float = Field(..., alias='Departamento_CHOCO')
    Departamento_CORDOBA: float = Field(..., alias='Departamento_CORDOBA')
    Departamento_CUNDINAMARCA: float = Field(..., alias='Departamento_CUNDINAMARCA')
    Departamento_GUAINIA: float = Field(..., alias='Departamento_GUAINIA')
    Departamento_GUAJIRA: float = Field(..., alias='Departamento_GUAJIRA')
    Departamento_GUAVIARE: float = Field(..., alias='Departamento_GUAVIARE')
    Departamento_HUILA: float = Field(..., alias='Departamento_HUILA')
    Departamento_LA_GUAJIRA: float = Field(..., alias='Departamento_LA GUAJIRA')
    Departamento_MAGDALENA: float = Field(..., alias='Departamento_MAGDALENA')
    Departamento_META: float = Field(..., alias='Departamento_META')
    Departamento_MONTERIA: float = Field(..., alias='Departamento_MONTERIA')
    Departamento_NARINO: float = Field(..., alias='Departamento_NARINO')
    Departamento_NORTE_DE_SANTANDER: float = Field(..., alias='Departamento_NORTE DE SANTANDER')
    Departamento_PUTUMAYO: float = Field(..., alias='Departamento_PUTUMAYO')
    Departamento_QUINDIO: float = Field(..., alias='Departamento_QUINDIO')
    Departamento_RISARALDA: float = Field(..., alias='Departamento_RISARALDA')
    Departamento_SAN_ANDRES_Y_PROVIDENCIA: float = Field(..., alias='Departamento_SAN ANDRES Y PROVIDENCIA')
    Departamento_SANTANDER: float = Field(..., alias='Departamento_SANTANDER')
    Departamento_SUCRE: float = Field(..., alias='Departamento_SUCRE')
    Departamento_TOLIMA: float = Field(..., alias='Departamento_TOLIMA')
    Departamento_VALLE: float = Field(..., alias='Departamento_VALLE')
    Departamento_VALLEDUPAR: float = Field(..., alias='Departamento_VALLEDUPAR')
    Departamento_VAUPES: float = Field(..., alias='Departamento_VAUPES')
    Departamento_VICHADA: float = Field(..., alias='Departamento_VICHADA')
    MACROSECTOR_AGROPECUARIO: float = Field(..., alias='MACROSECTOR_AGROPECUARIO')
    MACROSECTOR_COMERCIO: float = Field(..., alias='MACROSECTOR_COMERCIO')
    MACROSECTOR_CONSTRUCCION: float = Field(..., alias='MACROSECTOR_CONSTRUCCIÓN')
    MACROSECTOR_MANUFACTURA: float = Field(..., alias='MACROSECTOR_MANUFACTURA')
    MACROSECTOR_MINERO: float = Field(..., alias='MACROSECTOR_MINERO')
    MACROSECTOR_SERVICIOS: float = Field(..., alias='MACROSECTOR_SERVICIOS')
    CIIU_1_410: float = Field(..., alias='CIIU_1,410')
    CIIU_2_229: float = Field(..., alias='CIIU_2,229')
    CIIU_4_111: float = Field(..., alias='CIIU_4,111')
    CIIU_4_290: float = Field(..., alias='CIIU_4,290')
    CIIU_4_511: float = Field(..., alias='CIIU_4,511')
    CIIU_4_530: float = Field(..., alias='CIIU_4,530')
    CIIU_4_620: float = Field(..., alias='CIIU_4,620')
    CIIU_4_631: float = Field(..., alias='CIIU_4,631')
    CIIU_4_645: float = Field(..., alias='CIIU_4,645')
    CIIU_4_659: float = Field(..., alias='CIIU_4,659')
    CIIU_4_663: float = Field(..., alias='CIIU_4,663')
    CIIU_4_664: float = Field(..., alias='CIIU_4,664')
    CIIU_4_690: float = Field(..., alias='CIIU_4,690')
    CIIU_4_711: float = Field(..., alias='CIIU_4,711')
    CIIU_4_731: float = Field(..., alias='CIIU_4,731')
    CIIU_6_810: float = Field(..., alias='CIIU_6,810')
    CIIU_7_112: float = Field(..., alias='CIIU_7,112')
    CIIU_8_010: float = Field(..., alias='CIIU_8,010')
    CIIU_8_610: float = Field(..., alias='CIIU_8,610')
    CIIU_8_621: float = Field(..., alias='CIIU_8,621')
    CIIU_Other: float = Field(..., alias='CIIU_Other')

    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "INGRESOS OPERACIONALES": 0.5,
                "TOTAL ACTIVOS": 0.5,
                "TOTAL PASIVOS": 0.5,
                "TOTAL PATRIMONIO": 0.5,
                "Year": 2023,
                "TRM_Promedio": 4000,
                "IPC_Promedio": 5,
                "Inflacion_Promedio": 10,
                "Inversion_Extranjera_Total": 1000,
                "PIB_Departamental": 50000,
                "REGIÓN_Antioquia": 0,
                "REGIÓN_Bogotá - Cundinamarca": 1,
                "REGIÓN_Centro - Oriente": 0,
                "REGIÓN_Costa Atlántica": 0,
                "REGIÓN_Costa Pacífica": 0,
                "REGIÓN_Eje Cafetero": 0,
                "REGIÓN_Otros": 0,
                "Departamento_AMAZONAS": 0,
                "Departamento_BOGOTA": 1,
            }
        }

NAME_MAPPING = {
    "INGRESOS_OPERACIONALES": "INGRESOS OPERACIONALES",
    "TOTAL_ACTIVOS": "TOTAL ACTIVOS",
    "TOTAL_PASIVOS": "TOTAL PASIVOS",
    "TOTAL_PATRIMONIO": "TOTAL PATRIMONIO",
    "Year": "Year",
    "TRM_Promedio": "TRM_Promedio",
    "IPC_Promedio": "IPC_Promedio",
    "Inflacion_Promedio": "Inflacion_Promedio",
    "Inversion_Extranjera_Total": "Inversion_Extranjera_Total",
    "PIB_Departamental": "PIB_Departamental",
    "REGION_Antioquia": "REGIÓN_Antioquia",
    "REGION_Bogotá___Cundinamarca": "REGIÓN_Bogotá - Cundinamarca",
    "REGION_Centro___Oriente": "REGIÓN_Centro - Oriente",
    "REGION_Costa_Atlántica": "REGIÓN_Costa Atlántica",
    "REGION_Costa_Pacífica": "REGIÓN_Costa Pacífica",
    "REGION_Eje_Cafetero": "REGIÓN_Eje Cafetero",
    "REGION_Otros": "REGIÓN_Otros",
    "Departamento_AMAZONAS": "Departamento_AMAZONAS",
    "Departamento_ANTIOQUIA": "Departamento_ANTIOQUIA",
    "Departamento_ARAUCA": "Departamento_ARAUCA",
    "Departamento_ARMENIA": "Departamento_ARMENIA",
    "Departamento_ATLANTICO": "Departamento_ATLANTICO",
    "Departamento_BOGOTA": "Departamento_BOGOTA",
    "Departamento_BOLIVAR": "Departamento_BOLIVAR",
    "Departamento_BOYACA": "Departamento_BOYACA",
    "Departamento_CALDAS": "Departamento_CALDAS",
    "Departamento_CAQUETA": "Departamento_CAQUETA",
    "Departamento_CASANARE": "Departamento_CASANARE",
    "Departamento_CAUCA": "Departamento_CAUCA",
    "Departamento_CESAR": "Departamento_CESAR",
    "Departamento_CHOCO": "Departamento_CHOCO",
    "Departamento_CORDOBA": "Departamento_CORDOBA",
    "Departamento_CUNDINAMARCA": "Departamento_CUNDINAMARCA",
    "Departamento_GUAINIA": "Departamento_GUAINIA",
    "Departamento_GUAJIRA": "Departamento_GUAJIRA",
    "Departamento_GUAVIARE": "Departamento_GUAVIARE",
    "Departamento_HUILA": "Departamento_HUILA",
    "Departamento_LA_GUAJIRA": "Departamento_LA GUAJIRA",
    "Departamento_MAGDALENA": "Departamento_MAGDALENA",
    "Departamento_META": "Departamento_META",
    "Departamento_MONTERIA": "Departamento_MONTERIA",
    "Departamento_NARINO": "Departamento_NARINO",
    "Departamento_NORTE_DE_SANTANDER": "Departamento_NORTE DE SANTANDER",
    "Departamento_PUTUMAYO": "Departamento_PUTUMAYO",
    "Departamento_QUINDIO": "Departamento_QUINDIO",
    "Departamento_RISARALDA": "Departamento_RISARALDA",
    "Departamento_SAN_ANDRES_Y_PROVIDENCIA": "Departamento_SAN ANDRES Y PROVIDENCIA",
    "Departamento_SANTANDER": "Departamento_SANTANDER",
    "Departamento_SUCRE": "Departamento_SUCRE",
    "Departamento_TOLIMA": "Departamento_TOLIMA",
    "Departamento_VALLE": "Departamento_VALLE",
    "Departamento_VALLEDUPAR": "Departamento_VALLEDUPAR",
    "Departamento_VAUPES": "Departamento_VAUPES",
    "Departamento_VICHADA": "Departamento_VICHADA",
    "MACROSECTOR_AGROPECUARIO": "MACROSECTOR_AGROPECUARIO",
    "MACROSECTOR_COMERCIO": "MACROSECTOR_COMERCIO",
    "MACROSECTOR_CONSTRUCCION": "MACROSECTOR_CONSTRUCCIÓN",
    "MACROSECTOR_MANUFACTURA": "MACROSECTOR_MANUFACTURA",
    "MACROSECTOR_MINERO": "MACROSECTOR_MINERO",
    "MACROSECTOR_SERVICIOS": "MACROSECTOR_SERVICIOS",
    "CIIU_1_410": "CIIU_1,410",
    "CIIU_2_229": "CIIU_2,229",
    "CIIU_4_111": "CIIU_4,111",
    "CIIU_4_290": "CIIU_4,290",
    "CIIU_4_511": "CIIU_4,511",
    "CIIU_4_530": "CIIU_4,530",
    "CIIU_4_620": "CIIU_4,620",
    "CIIU_4_631": "CIIU_4,631",
    "CIIU_4_645": "CIIU_4,645",
    "CIIU_4_659": "CIIU_4,659",
    "CIIU_4_663": "CIIU_4,663",
    "CIIU_4_664": "CIIU_4,664",
    "CIIU_4_690": "CIIU_4,690",
    "CIIU_4_711": "CIIU_4,711",
    "CIIU_4_731": "CIIU_4,731",
    "CIIU_6_810": "CIIU_6,810",
    "CIIU_7_112": "CIIU_7,112",
    "CIIU_8_010": "CIIU_8,010",
    "CIIU_8_610": "CIIU_8,610",
    "CIIU_8_621": "CIIU_8,621",
    "CIIU_Other": "CIIU_Other"
}

@app.get("/")
def read_root():
    return {"message": "Welcome to the Earnings Prediction API. Visit /docs for documentation."}

@app.post("/predict")
def predict(input_data: PredictionInput):
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Convert input to DataFrame with correct column names and order
    data_dict = input_data.dict(by_alias=True)
    
    # Ensure correct order of columns as expected by the model
    ordered_columns = list(NAME_MAPPING.values())
    
    input_df = pd.DataFrame([data_dict])
    
    # Reorder columns just to be safe
    input_df = input_df[ordered_columns]
    
    # Convert to DMatrix for Booster
    dmatrix = xgb.DMatrix(input_df)
    
    try:
        prediction = model.predict(dmatrix)
        # Booster.predict returns a numpy array
        return {"prediction": float(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
