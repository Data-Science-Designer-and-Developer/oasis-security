# generate_model.py - 1-click model generation
from model import CrimeRatePredictor
import os

# Créer dossier si absent
os.makedirs("../../models", exist_ok=True)

predictor = CrimeRatePredictor()
metrics = predictor.train("https://static.data.gouv.fr/.../delinquance.csv")
predictor.save("../../models/crime_predictor.pkl")

print("✅ Modèle créé : models/crime_predictor.pkl")
print(f"📊 R²: {metrics['r2_score']:.3f}")