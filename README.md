🛡️ Oasis Security — Crime Predictor








Predictive model for crime rates in France (per 100,000 inhabitants)
Full ML pipeline: data collection → cleaning → modeling → API → dashboard
Official source: SSMSI / data.gouv.fr
 — 2016-2023

📋 Table of Contents
Context & Goals
Project Structure
Installation & Run
Data Pipeline
Modeling & Results
Streamlit Dashboard
FastAPI
Tests
Docker & CI/CD
Ethics & Limitations

🎯 Context & Goals

This project predicts French departmental crime rates by category, using official data from the national police and gendarmerie.

Primary use case: statistical exploration tool for journalists, social science researchers, and public policy decision-makers.

Technical objectives:

Build a fully reproducible end-to-end ML pipeline
Compare multiple regression algorithms with MLflow tracking
Deploy a prediction API (FastAPI) and interactive dashboard (Streamlit)
Apply MLOps best practices: versioning, testing, CI/CD, Docker

📁 Project Structure
oasis-security/ 
├── .github/ 
│   └── workflows/           # GitHub Actions CI/CD
├── data/                    # Cleaned data (.parquet)
├── docs/ 
│   └── crime_predictor/     # Technical documentation
├── images/                  # Visualizations & plots
├── models/ 
│   └── crime_predictor/ 
│       ├── src/ 
│       │   ├── train.py     # ← Training pipeline (model comparison)
│       │   └── predict.py   # ← FastAPI API
│       ├── models/ 
│       │   ├── crime_predictor.pkl    # Serialized model
│       │   └── metrics.json           # Train/test metrics
│       ├── mlruns/          # MLflow experiments
│       └── tests/ 
│           └── test_model.py   # ← Unit tests
├── notebooks/               # Exploration & EDA
├── pipeline/                # Automation scripts
├── streamlit/               # Additional Streamlit assets
├── app.py                   # ← Main Streamlit dashboard
├── script_crimes_et_delits.py   # ← Data collection & cleaning
├── Dockerfile               # Multi-stage build (train → production)
├── docker-compose.yml       # Full stack (MLflow + Postgres + API)
├── requirements.txt 
└── README.md 

⚙️ Installation & Run
1. Clone & Install
git clone https://github.com/Data-Science-Designer-and-Developer/oasis-security.git
cd oasis-security
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

2. Download & Clean Data
python script_crimes_et_delits.py
# → generates data/crimes_clean.parquet

3. Train the Model
python models/crime_predictor/src/train.py
# → compares 4 models, logs in MLflow, saves the best model
# → generates models/crime_predictor/models/crime_predictor.pkl
# → generates models/crime_predictor/models/metrics.json

4. Launch the Dashboard
streamlit run app.py
# → http://localhost:8501

5. Launch the API
uvicorn models.crime_predictor.src.predict:app --reload --port 8000
# → http://localhost:8000/docs

🔄 Data Pipeline
data.gouv.fr (SSMSI)
        ↓
script_crimes_et_delits.py
        ├── CSV download (requests)
        ├── Normalize column names (snake_case)
        ├── Remove duplicates
        ├── Cast numeric types
        ├── Remove abnormal rates (< 0)
        ├── Feature engineering
        │   ├── annual_rate_change (% change per dep × category)
        │   └── normalized_year ([0, 1])
        └── Save as Parquet (Snappy)
                ↓
        data/crimes_clean.parquet

Raw data: 8 columns, ~50,000 rows
After cleaning: 10 columns, ~49,000 rows (< 2% loss)

🤖 Modeling & Results
Features used
| Feature       | Description                   |
| ------------- | ----------------------------- |
| `annee`       | Year (int)                    |
| `dep_encoded` | Department (LabelEncoded)     |
| `cat_encoded` | Crime category (LabelEncoded) |
| `annee_norm`  | Normalized year [0, 1]        |

Target: tauxpour100000hab (crime rate per 100,000 inhabitants)
Split: 80% train / 20% test — seed 42
Validation: K-Fold cross-validation (k=5) on training set

Model Comparison (Test Set)
| Model             | R² test  | RMSE     | MAE      | CV R² (±std)    |
| ----------------- | -------- | -------- | -------- | --------------- |
| Ridge             | 0.71     | 87.4     | 62.1     | 0.69 ± 0.03     |
| Random Forest     | 0.89     | 54.2     | 38.7     | 0.87 ± 0.02     |
| Gradient Boosting | 0.88     | 56.1     | 40.2     | 0.86 ± 0.02     |
| **XGBoost** ✅    | **0.91** | **49.8** | **35.3** | **0.90 ± 0.01** |

Best model: XGBoost — R²=0.91 on test set
Small train/test gap → no significant overfitting
Low CV variance → confirmed robustness

MLflow Tracking
mlflow ui --backend-store-uri models/crime_predictor/mlruns
# → http://localhost:5000

📊 Streamlit Dashboard

5 interactive pages:
| Page                 | Content                                               |
| -------------------- | ----------------------------------------------------- |
| Overview             | KPIs, boxplots by category, top 10 departments        |
| Department Analysis  | Multi-department comparison, heatmap                  |
| Temporal Trends      | Evolution 2016-2023, base 100 index, annual variation |
| ML Prediction        | Interactive simulator with historical chart           |
| Ethics & Limitations | Documentation of biases and usage limits              |

🌐 FastAPI
Endpoints
| Method | Endpoint   | Description                       |
| ------ | ---------- | --------------------------------- |
| GET    | `/health`  | API status + model metrics        |
| POST   | `/predict` | Predict crime rate                |
| GET    | `/docs`    | Interactive Swagger documentation |

Example Request
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"annee": 2025, "dep_encoded": 5, "cat_encoded": 0, "annee_norm": 1.0}'
  {
  "predicted_rate": 312.47,
  "unit": "crimes per 100,000 inhabitants",
  "model_used": "XGBoost",
  "r2_test": 0.91
}

🧪 Tests
# Run all tests
pytest models/crime_predictor/tests/ -v

# With coverage
pytest models/crime_predictor/tests/ -v --cov=models/crime_predictor/src --cov-report=term-missing

Test Coverage:
| Class               | Tests                                                       |
| ------------------- | ----------------------------------------------------------- |
| `TestData`          | DataFrame integrity (6 assertions)                          |
| `TestModel`         | Shape, type, positivity, R², determinism (7 assertions)     |
| `TestSerialization` | Joblib serialization, metrics.json structure (2 assertions) |

🐳 Docker & CI/CD
Multi-stage Docker
# Build (trainer → production)
docker build -t oasis-security:latest .

# Run the API
docker run -p 8000:8000 oasis-security:latest

Full Stack (MLflow + Postgres + API)
docker-compose up -d
# MLflow UI  → http://localhost:5000
# API        → http://localhost:8000/docs

GitHub Actions CI/CD

Workflow .github/workflows/ci-cd.yml runs on each push:

Lint (flake8)
Unit tests (pytest)
Docker build
Push image to GHCR

⚠️ Ethics & Limitations

This model is a statistical exploration tool, not an operational decision system.

Data limitations:

Only includes recorded offenses (dark figure estimated 50-80%)
Recording practices vary across services
No infra-departmental data

Model biases:

Reflects inherent reporting biases
Correlation ≠ causation
Not suited for exogenous shocks (COVID, economic crises)

Prohibited uses:

Predictive targeting of individuals or geographic areas
Judicial or law enforcement decision support

Compliance: aggregated anonymized open data — no personal data.

📜 License
MIT — see LICENSE

👤 Author
Frédéric Tellier — Data Scientist
LinkedIn: https://www.linkedin.com/in/fr%C3%A9d%C3%A9ric-tellier-8a9170283/ ; Portfolio: https://github.com/Dreipfelt/

Project completed as part of the CDSD certification — 2026
