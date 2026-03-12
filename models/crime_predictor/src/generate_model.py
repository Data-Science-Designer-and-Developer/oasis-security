# generate_model.py
import os
from model import CrimeRatePredictor

# Créer dossier models/
os.makedirs("../../models", exist_ok=True)

# Générer modèle
predictor = CrimeRatePredictor()
metrics = predictor.train("https://data.gouv.fr")
predictor.save("../../models/crime_predictor.pkl")

print("🎉 MODÈLE CRÉÉ!")
print(f"📁 models/crime_predictor.pkl → PRÊT!")
