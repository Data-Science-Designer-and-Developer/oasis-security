# oasis/models/crime_predictor/train.py
import mlflow
import mlflow.lightgbm
import argparse
import pandas as pd
from model import CrimeRatePredictor
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-url', required=True, help='URL data.gouv.fr')
    parser.add_argument('--experiment-name', default='crime_predictor')
    parser.add_argument('--run-name', default='lightgbm_v1')
    parser.add_argument('--model-path', default='models/crime_predictor.pkl')
    args = parser.parse_args()
    
    # MLflow setup
    mlflow.set_experiment(args.experiment_name)
    mlflow.set_tracking_uri("http://localhost:5000")  # ou MLflow server
    
    with mlflow.start_run(run_name=args.run_name):
        # Initialiser modèle
        predictor = CrimeRatePredictor()
        
        # Entraîner
        metrics = predictor.train(args.data_url)
        
        # Log paramètres
        mlflow.log_params(predictor.config['model'])
        mlflow.log_metrics({
            'r2_score': metrics['r2_score'],
            'n_features': len(predictor.FEATURE_COLS)
        })
        
        # Log feature importance
        for feature, importance in metrics['feature_importance'].items():
            mlflow.log_metric(f"feature_importance_{feature}", importance)
        
        # Log modèle MLflow
        mlflow.lightgbm.log_model(
            predictor.model,
            "model",
            input_example=pd.DataFrame({f: [0]*10 for f in predictor.FEATURE_COLS}).head(1)
        )
        
        # Sauvegarde locale
        Path(args.model_path).parent.mkdir(parents=True, exist_ok=True)
        predictor.save(args.model_path)
        
        logger.info(f"✅ Modèle sauvegardé: {args.model_path}")
        logger.info(f"📊 R²: {metrics['r2_score']:.3f}")

if __name__ == "__main__":
    main()
