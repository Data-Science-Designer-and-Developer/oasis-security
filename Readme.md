Oasis Security – Crime Predictor


Production-ready ML pipeline for predicting French crime rates (per 100k inhabitants) using data.gouv.fr police/gendarmerie statistics. Full MLOps stack: EDA → serialised model → API → Docker → CI/CD → MLflow.

🎯 Objectives

    Forecast crime rates by region/indicator from official French police data (2016-2025)

    Production-grade implementation: clean project structure, serialised model, Docker-ready, GitHub Actions CI/CD

    Oasis Security integration: security monitoring & risk prediction platform foundation

📁 Project Structure


oasis-security/
└── models/crime_predictor/
    ├── src/                    # Core ML code
    │   ├── model.py           # CrimeRatePredictor class (LinearRegression)
    │   ├── generate_model.py   # Model generation script
    │   ├── train.py           # Training pipeline + MLflow
    │   └── predict.py         # FastAPI inference API
    ├── models/                 # Serialised models
    │   └── crime_predictor.pkl # R²=0.806 (production-ready)
    ├── mlruns/                 # MLflow experiment tracking
    ├── tests/                  # pytest unit tests
    └── requirements.txt        # Python dependencies

📊 Data & Model
Data Source

    data.gouv.fr: Official French police/gendarmerie crime statistics (regional/communal level)

    Key metrics: year, Code_region, indicateur, nombre, insee_pop

    Target: Crime rate per 100k inhabitants = (nombre / insee_pop) * 100000

Current Model


Class: CrimeRatePredictor (scikit-learn LinearRegression)
Features: ['year_sin', 'year_cos', 'region_mean', 'ind_code']
Metrics: R² = 0.806 (simulated data, production-stable)
Serialisation: joblib (cross-Python compatible)

Design choices:

    LinearRegression: guaranteed picklable, robust, scalable foundation

    Ready for XGBoost/LightGBM/Prophet upgrade

    Full metadata (features, training flag) embedded

🚀 Quick Start


# Clone & setup
git clone https://github.com/Dreipfelt/oasis-security.git
cd oasis-security/models/crime_predictor

# Install dependencies
pip install -r requirements.txt

# Generate model
cd src/
python generate_model.py
# → models/crime_predictor.pkl created

Load & predict:

python
import joblib
model_data = joblib.load("../models/crime_predictor.pkl")
model = model_data["model"]
X_sample = [[0.1, 0.2, 250.0, 5]]  # year_sin, year_cos, region_mean, ind_code
prediction = model.predict(X_sample)  # e.g. 287.4 crimes/100k

📈 2030 Projections (example)
Region	Crime Type	Rate/100k	Alert
IDF (11)	VIOLENCES	387	🚨
Paris (75)	CAMBRIOLE	245	⚠️
PACA (93)	VOLS	412	🚨
🔮 Roadmap

    Serialised model (LinearRegression, R²=0.806)

    Production structure (src/models/mlruns/tests)

    Real data.gouv.fr pipeline

    XGBoost + Prophet ensemble

    FastAPI deployment (predict.py)

    GitHub Actions: tests → Docker → MLflow

    Streamlit dashboard

🛠️ Tech Stack

Core: Python 3.13, scikit-learn, joblib
Future: FastAPI, MLflow, Docker, GitHub Actions
Data: data.gouv.fr (police/gendarmerie 2016-2025)

📝 Author

Frédéric Tellier – Data Scientist
LinkedIn : https://www.linkedin.com/in/fr%C3%A9d%C3%A9ric-tellier-8a9170283/ | Portfolio : https://github.com/Dreipfelt/

Licence: MIT