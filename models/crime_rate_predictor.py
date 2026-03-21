# models/crime_rate_predictor.py
import pandas as pd
import lightgbm as lgb
import numpy as np
from typing import Dict, Any
from pydantic import BaseModel

class CrimeRatePredictor:
    def __init__(self):
        self.model = None
        self.feature_cols = [
            'year_sin', 'year_cos', 'lag1', 'lag2', 
            'region_mean', 'ind_code', 'reg_code'
        ]
    
    def load_data(self, url: str) -> pd.DataFrame:
        """Charge et nettoie données délinquance"""
        df = pd.read_csv(url, sep=';')
        df = df[df['unite_de_compte'] == 'nombre'].copy()
        df['taux_100k'] = df['nombre'] / df['insee_pop'] * 100000
        return df.dropna(subset=['taux_100k'])
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Features production-ready"""
        df = df.copy()
        df['year_sin'] = np.sin(2 * np.pi * df['annee']/10)
        df['year_cos'] = np.cos(2 * np.pi * df['annee']/10)
        df['lag1'] = df['taux_100k'].shift(1).fillna(df['taux_100k'].mean())
        df['lag2'] = df['taux_100k'].shift(2).fillna(df['taux_100k'].mean())
        df['region_mean'] = df.groupby('Code_region')['taux_100k'].transform('mean')
        df['ind_code'] = pd.factorize(df['indicateur'])[0]
        df['reg_code'] = df['Code_region'].astype(int)
        return df.fillna(0)
    
    def train(self, df: pd.DataFrame):
        """Entraînement LightGBM production"""
        df_features = self.engineer_features(df)
        X = df_features[self.feature_cols]
        y = df_features['taux_100k']
        
        self.model = lgb.LGBMRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.08,
            num_leaves=50, random_state=42, verbose=-1, n_jobs=1
        )
        self.model.fit(X, y)
        return self.model.feature_importances_
    
    def predict_2030(self, indicateur: str, region: str, df: pd.DataFrame) -> float:
        """Prédiction 2030 pour oasis API"""
        df_f = self.engineer_features(df)
        mask = (df_f['indicateur'] == indicateur) & (df_f['Code_region'] == region)
        if mask.sum() == 0:
            return np.nan
        
        last_row = df_f[mask].iloc[-1][self.feature_cols].copy()
        last_row['annee'] = 2030  # Future
        last_row['year_sin'] = np.sin(2 * np.pi * 2030/10)
        last_row['year_cos'] = np.cos(2 * np.pi * 2030/10)
        
        return self.model.predict(last_row.values.reshape(1,-1))[0]