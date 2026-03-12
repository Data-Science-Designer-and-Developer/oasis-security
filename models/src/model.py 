# oasis/models/crime_predictor/model.py
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
from typing import Dict, Any, Optional
import yaml
from pathlib import Path

class CrimeRatePredictor:
    """Modèle LightGBM production pour prédiction délinquance"""
    
    FEATURE_COLS = [
        'year_sin', 'year_cos', 'year_trend', 'lag1', 'lag2', 
        'roll_mean_3', 'region_mean', 'ind_code', 'reg_code'
    ]
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = self._load_config(config_path)
        self.model = None
        self.feature_names = self.FEATURE_COLS
        self.scaler = None
    
    def _load_config(self, config_path: str) -> Dict:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def load_data(self, url: str) -> pd.DataFrame:
        """Charge et nettoie données data.gouv.fr"""
        df = pd.read_csv(url, sep=';')
        df = df[df['unite_de_compte'] == 'nombre'].copy()
        df['taux_100k'] = df['nombre'] / df['insee_pop'] * 100_000
        return df.dropna(subset=['taux_100k'])
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Feature engineering senior-level"""
        df = df.copy()
        
        # Features temporelles cycliques
        df['year_sin'] = np.sin(2 * np.pi * df['annee'] / 10)
        df['year_cos'] = np.cos(2 * np.pi * df['annee'] / 10)
        df['year_trend'] = (df['annee'] - df['annee'].min()) / (df['annee'].max() - df['annee'].min())
        
        # Lags et rolling
        df['lag1'] = df.groupby(['indicateur', 'Code_region'])['taux_100k'].shift(1).fillna(method='bfill')
        df['lag2'] = df.groupby(['indicateur', 'Code_region'])['taux_100k'].shift(2).fillna(method='bfill')
        df['roll_mean_3'] = df.groupby(['indicateur', 'Code_region'])['taux_100k'].rolling(3, min_periods=1).mean().reset_index(0, drop=True)
        
        # Features agrégées
        df['region_mean'] = df.groupby('Code_region')['taux_100k'].transform('mean')
        
        # Encoding
        df['ind_code'] = pd.Categorical(df['indicateur']).codes
        df['reg_code'] = pd.Categorical(df['Code_region']).codes
        
        return df[self.FEATURE_COLS + ['taux_100k']].dropna()
    
    def train(self, data_url: str):
        """Entraînement avec validation temporelle"""
        print("📊 Chargement données...")
        df = self.load_data(data_url)
        
        print("🔧 Feature engineering...")
        df_features = self.engineer_features(df)
        
        X = df_features[self.FEATURE_COLS]
        y = df_features['taux_100k']
        
        # TimeSeriesSplit validation
        tscv = TimeSeriesSplit(n_splits=3)
        scores = []
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            model = lgb.LGBMRegressor(
                n_estimators=self.config['model']['n_estimators'],
                max_depth=self.config['model']['max_depth'],
                learning_rate=self.config['model']['learning_rate'],
                num_leaves=self.config['model']['num_leaves'],
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
            model.fit(X_train, y_train)
            
            score = model.score(X_val, y_val)
            scores.append(score)
        
        print(f"✅ Validation R²: {np.mean(scores):.3f} ± {np.std(scores):.3f}")
        
        # Modèle final
        self.model = lgb.LGBMRegressor(**self.config['model'])
        self.model.fit(X, y)
        
        return {
            'r2_score': self.model.score(X, y),
            'feature_importance': dict(zip(self.FEATURE_COLS, self.model.feature_importances_))
        }
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Prédiction batch"""
        if self.model is None:
            raise ValueError("Modèle non entraîné. Appelez train() d'abord.")
        return self.model.predict(X[self.FEATURE_COLS])
    
    def save(self, path: str):
        """Sauvegarde modèle + metadata"""
        joblib.dump({
            'model': self.model,
            'feature_names': self.FEATURE_COLS,
            'config': self.config
        }, path)
    
    def load(self, path: str):
        """Chargement modèle"""
        data = joblib.load(path)
        self.model = data['model']
        self.feature_names = data['feature_names']
        self.config = data['config']
