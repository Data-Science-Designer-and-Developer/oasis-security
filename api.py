"""
Oasis Security – Crime Predictor API
FastAPI + MLflow production-ready endpoint
"""

from contextlib import asynccontextmanager
from typing import Dict, Optional

import mlflow
import mlflow.lightgbm
import numpy as np
import os
import pandas as pd
import uvicorn

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Config MLflow
# ---------------------------------------------------------------------------
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment("crime_predictor_prod")

# ---------------------------------------------------------------------------
# Lifespan : chargement modèle au démarrage
# ---------------------------------------------------------------------------
predictor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    print("🚀 Chargement modèle...")
    try:
        model_uri = "models:/crime_predictor_prod/Production"
        predictor = mlflow.lightgbm.load_model(model_uri)
        print("✅ Modèle chargé depuis MLflow Registry")
    except Exception:
        # Fallback : modèle local sérialisé
        from models.crime_predictor.src.model import CrimeRatePredictor
        predictor = CrimeRatePredictor()
        predictor.load("models/crime_predictor/artifacts/crime_predictor.pkl")
        print("✅ Modèle chargé depuis fichier local")
    yield
    print("🛑 API shutdown")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Oasis Security – Crime Predictor API",
    version="2.0.0",
    description="Prédiction du taux de délinquance par région (pour 100 000 habitants)",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Schémas
# ---------------------------------------------------------------------------
class PredictionRequest(BaseModel):
    year: int = 2030
    indicateur: str
    region: str
    lag1: Optional[float] = 250.0
    lag2: Optional[float] = 245.0

    model_config = {"json_schema_extra": {
        "example": {
            "year": 2030,
            "indicateur": "Coups et blessures volontaires",
            "region": "R11",
            "lag1": 280.5,
            "lag2": 275.0,
        }
    }}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", tags=["Monitoring"])
async def health():
    """Vérifie que l'API et le modèle sont opérationnels."""
    return {
        "status": "healthy",
        "model_loaded": predictor is not None,
        "model_version": "v2.0",
        "mlflow_uri": MLFLOW_URI,
    }


@app.post("/predict", response_model=Dict, tags=["Prédiction"])
async def predict(request: PredictionRequest):
    """
    Prédit le taux de délinquance pour un indicateur et une région donnés.

    - **year** : année cible (ex. 2030)
    - **indicateur** : catégorie de crime (ex. "Coups et blessures volontaires")
    - **region** : code région INSEE (ex. "R11" pour Île-de-France)
    - **lag1 / lag2** : taux des 2 années précédentes (optionnel, valeurs par défaut utilisées si absent)
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")

    with mlflow.start_run(nested=True) as run:
        try:
            lag1 = request.lag1 or 250.0
            lag2 = request.lag2 or 245.0

            features = pd.DataFrame([{
                "year_sin":    np.sin(2 * np.pi * request.year / 10),
                "year_cos":    np.cos(2 * np.pi * request.year / 10),
                "year_trend":  (request.year - 2016) / 9,
                "lag1":        lag1,
                "lag2":        lag2,
                "roll_mean_3": (lag1 + lag2 + 240.0) / 3,
                "region_mean": 250.0,
                "ind_code":    hash(request.indicateur) % 100,
                "reg_code":    int(request.region.replace("R", "")),
            }])

            pred = float(predictor.predict(features)[0])

            # Observabilité MLflow
            mlflow.log_params({
                "indicateur": request.indicateur,
                "region":     request.region,
                "year":       request.year,
            })
            mlflow.log_metric("prediction", pred)

            niveau = (
                "🚨 Risque élevé"   if pred > 400 else
                "⚠️  Risque modéré" if pred > 300 else
                "✅ Risque faible"
            )

            return {
                "prediction":    round(pred, 2),
                "unit":          "taux / 100 000 habitants",
                "year":          request.year,
                "indicateur":    request.indicateur,
                "region":        request.region,
                "interpretation": niveau,
                "mlflow_run_id": run.info.run_id,
            }

        except Exception as e:
            mlflow.log_metric("error", 1)
            raise HTTPException(status_code=500, detail=str(e))


@app.get("/leaderboard", tags=["Analyse"])
async def leaderboard():
    """
    Retourne le top 5 des combinaisons région/indicateur
    avec les prédictions 2030 les plus élevées (risques prioritaires).
    """
    try:
        client = mlflow.MlflowClient()
        runs = client.search_runs(
            experiment_ids=["0"],
            order_by=["metrics.prediction DESC"],
            max_results=50,
        )
        return {
            "top_risks": [
                {
                    "indicateur": r.data.params.get("indicateur", "N/A"),
                    "region":     r.data.params.get("region", "N/A"),
                    "pred_2030":  r.data.metrics.get("prediction", 0),
                }
                for r in runs
            ][:5]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Lancement direct
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
