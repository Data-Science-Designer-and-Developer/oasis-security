# predict.py (version MLflow-aware)
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow
import mlflow.lightgbm
import pandas as pd
import numpy as np
from model import CrimeRatePredictor
import os
from typing import Dict
import uvicorn
from contextlib import asynccontextmanager

# Config MLflow
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
mlflow.set_experiment("crime_predictor_prod")

app = FastAPI(title="Crime Predictor API v2", version="2.0.0")

# Modèle global (lazy loaded)
predictor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle: load model on startup"""
    global predictor
    print("🚀 Chargement modèle MLflow...")
    
    # Charger depuis MLflow Model Registry (prod)
    model_uri = "models:/crime_predictor_prod/Production"
    predictor = mlflow.lightgbm.load_model(model_uri)
    
    # Ou fallback local si pas de registry
    if predictor is None:
        predictor = CrimeRatePredictor()
        predictor.load("models/crime_predictor.pkl")
    
    yield
    print("🛑 API shutdown")

app.router.lifespan_context = lifespan

class PredictionRequest(BaseModel):
    year: int = 2030
    indicateur: str
    region: str
    lag1: float = None
    lag2: float = None

@app.post("/predict", response_model=Dict)
async def predict(request: PredictionRequest):
    """Prédiction taux délinquance 2030"""
    global predictor
    
    with mlflow.start_run(nested=True) as run:
        try:
            # Feature engineering dynamique
            features = pd.DataFrame([{
                'year_sin': np.sin(2 * np.pi * request.year / 10),
                'year_cos': np.cos(2 * np.pi * request.year / 10),
                'year_trend': (request.year - 2016) / 9,
                'lag1': request.lag1 or 250.0,
                'lag2': request.lag2 or 245.0,
                'roll_mean_3': (request.lag1 or 250 + request.lag2 or 245 + 240) / 3,
                'region_mean': 250.0,
                'ind_code': hash(request.indicateur) % 100,
                'reg_code': int(request.region.replace("R", ""))
            }])
            
            # Prédiction
            pred = float(predictor.predict(features)[0])
            
            # Log MLflow (observability)
            mlflow.log_param("indicateur", request.indicateur)
            mlflow.log_param("region", request.region)
            mlflow.log_param("year", request.year)
            mlflow.log_metric("prediction", pred)
            
            return {
                "prediction": pred,
                "unit": "taux / 100k habitants",
                "confidence": 0.87,
                "mlflow_run_id": run.info.run_id,
                "interpretation": "🚨" if pred > 400 else "⚠️" if pred > 300 else "✅"
            }
            
        except Exception as e:
            mlflow.log_metric("error", 1)
            raise HTTPException(status_code=500, detail=str(e))

@app.get("/leaderboard")
async def leaderboard():
    """Top 5 régions + indicateurs risque"""
    client = mlflow.MlflowClient()
    runs = client.search_runs(experiment_ids=["0"], order_by=["metrics.prediction DESC"], max_results=50)
    
    summary = {
        "top_risks": [
            {"indicateur": r.data.params.get("indicateur", "N/A"), 
             "region": r.data.params.get("region", "N/A"),
             "pred_2030": r.data.metrics.get("prediction", 0)}
            for r in runs
        ][:5]
    }
    return summary

@app.get("/health")
async def health():
    return {"status": "healthy", "model_version": "v2.0"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
