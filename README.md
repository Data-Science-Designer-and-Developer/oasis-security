# 🛡️ Oasis Security — Crime Predictor

[![CI/CD](https://github.com/Data-Science-Designer-and-Developer/oasis-security/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/Data-Science-Designer-and-Developer/oasis-security/actions)
[![Docker](https://img.shields.io/badge/Docker-GHCR-blue)](https://ghcr.io/Data-Science-Designer-and-Developer)
[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

> **Predictive model for crime rates in France** (per 100,000 inhabitants)  
> Full ML pipeline: data collection → cleaning → modeling → API → dashboard  
> Official source: SSMSI / [data.gouv.fr](https://www.data.gouv.fr) — 2016–2023

---

## 📋 Table of Contents

1. [Context & Goals](#-context--goals)
2. [Project Structure](#-project-structure)
3. [Installation & Running](#-installation--running)
4. [Data Pipeline](#-data-pipeline)
5. [Modeling & Results](#-modeling--results)
6. [Streamlit Dashboard](#-streamlit-dashboard)
7. [FastAPI Endpoints](#-fastapi-endpoints)
8. [Tests](#-tests)
9. [Docker & CI/CD](#-docker--cicd)
10. [Ethics & Limitations](#-ethics--limitations)

---

## 🎯 Context & Goals

This project predicts **French departmental crime rates** by category using official police and gendarmerie data.

**Primary use case**: a statistical exploration tool for journalists, social science researchers, and public policy makers.

**Technical objectives**:

- Build a fully reproducible end-to-end ML pipeline
- Compare multiple regression algorithms using MLflow tracking
- Deploy a prediction API (FastAPI) and interactive dashboard (Streamlit)
- Follow MLOps best practices: versioning, testing, CI/CD, Docker

---

## 📁 Project Structure


oasis-security/
├── .github/
│ └── workflows/ # GitHub Actions CI/CD
├── data/ # Cleaned datasets (.parquet)
├── docs/
│ └── crime_predictor/ # Technical documentation
├── images/ # Visualizations and plots
├── models/
│ └── crime_predictor/
│ ├── src/
│ │ ├── train.py # Training pipeline (model comparison)
│ │ └── predict.py # FastAPI endpoint definitions
│ ├── models/
│ │ ├── crime_predictor.pkl # Serialized model
│ │ └── metrics.json # Train/test metrics
│ ├── mlruns/ # MLflow experiment tracking
│ └── tests/
│ └── test_model.py # Unit tests
├── notebooks/ # Exploration & EDA notebooks
├── pipeline/ # Automation scripts
├── streamlit/ # Streamlit supplementary assets
├── app.py # Main Streamlit dashboard
├── script_crimes_et_delits.py # Data collection & cleaning
├── Dockerfile # Multi-stage build (train → production)
├── docker-compose.yml # Full stack (MLflow + Postgres + API)
├── requirements.txt
└── README.md


---

## ⚙️ Installation & Running

### 1. Clone & Install Dependencies

```bash
git clone https://github.com/Data-Science-Designer-and-Developer/oasis-security.git
cd oasis-security
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
2. Download and Clean Data
python script_crimes_et_delits.py
# → generates data/crimes_clean.parquet
3. Train the Model
python models/crime_predictor/src/train.py
# → compares 4 models, logs to MLflow, saves best model
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
        ├── Download CSV (requests)
        ├── Normalize column names (snake_case)
        ├── Remove duplicates
        ├── Convert numeric types
        ├── Remove outlier rates (<0)
        ├── Feature engineering
        │   ├── annual_rate_change (pct_change by dep × category)
        │   └── year_norm (normalized [0, 1])
        └── Save Parquet (Snappy)
                ↓
        data/crimes_clean.parquet

Raw data: 8 columns, ~50,000 rows
After cleaning: 10 columns, ~49,000 rows (<2% loss)

🤖 Modeling & Results
Features
Feature	Description
annee	Year (int)

Project developed as part of CDSD certification — 2025
