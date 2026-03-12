
import joblib
model = joblib.load("model.joblib")

def predict(annee,ville_enc,nb):
    return model.predict([[annee,ville_enc,nb]])[0]
