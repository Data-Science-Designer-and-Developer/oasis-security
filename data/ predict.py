# oasis/models/crime_predictor/predict.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from model import CrimeRatePredictor
from typing import List, Dict
import uvicorn

app = FastAPI(title="Crime Predictor API", version="1.0.0")

# Charger modèle au démarrage
predictor = CrimeRatePredictor()
predictor.load("models/crime_predictor.pkl")

class PredictionRequest(BaseModel):
    year: int
    indicateur: str
    region: str
    lag1: float = 0.0
    lag2: float = 0.0

@app.post("/predict", response_model=Dict)
async def predict(request: PredictionRequest):
    """Prédiction taux délinquance"""
    try:
        # Créer features
        features = pd.DataFrame([{
            'year_sin': np.sin(2 * np.pi * request.year / 10),
            'year_cos': np.cos(2 * np.pi * request.year / 10),
            'year_trend': (request.year - 2016) / (2025 - 2016),
            'lag1': request.lag1,
            'lag2': request.lag2,
            'roll_mean_3': (request.lag1 + request.lag2 + 0) / 3,
            'region_mean': 250.0,  # Default
            'ind_code': hash(request.indicateur) % 100,
            'reg_code': int(request.region)
        }])
        
        pred = predictor.predict(features)[0]
        return {
            "prediction": float(pred),
            "confidence": 0.85,  # À calculer
            "unit": "taux / 100k habitants"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
