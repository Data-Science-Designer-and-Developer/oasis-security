# 🏴‍☠️ Oasis Security – Crime Predictor

[![CI/CD](https://github.com/Data-Science-Designer-and-Developer/oasis-security/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/Data-Science-Designer-and-Developer/oasis-security/actions)
[![Docker](https://img.shields.io/badge/Docker-GHCR-blue)](https://ghcr.io/Data-Science-Designer-and-Developer)
[![Model](https://img.shields.io/badge/R²-0.806-brightgreen)](https://github.com/Data-Science-Designer-and-Developer/oasis-security/blob/main/models/crime_predictor/models/crime_predictor.pkl)

A predictive model for crime rates in France (per 100,000 inhabitants), fully structured for production use.  
This repository demonstrates a **ML project with clear structure, documentation, modeling pipeline, and deployment readiness**.
---

## 🚀 Project Objectives

- Build a reliable predictive model using official French crime data (from data.gouv.fr).
- Establish a **clean, reproducible MLOps pipeline**:
  - Clear folder structure (`data/`, `notebooks/`, `models/`, etc.)
  - Serialized model for reuse
  - Training & inference scripts
  - Integrated CI/CD workflows

---

## 📁 Repository Structure

oasis-security/
├── .github/ # GitHub workflows (CI/CD)
├── data/ # Processed data files
├── docs/ # Documentation & dashboards
├── images/ # Visual assets & plots
├── models/
│ └── crime_predictor/
│ ├── src/ # Source code for model
│ ├── models/ # Serialized model (.pkl)
│ ├── mlruns/ # MLflow tracking data
│ ├── tests/ # Unit tests (optional)
│ └── requirements.txt # Dependencies for this model
├── notebooks/ # Exploration & analysis notebooks
├── pipeline/ # Scripts for automation
├── Dockerfile # Docker configuration
├── LICENSE # License
└── README.md # Project overview


---

## 📊 Usage

1. Create & activate a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate

2. Install dependencies:
'''bash 
pip install -r models/crime_predictor/requirements.txt

3. Run training: 
'''bash
python models/crime_predictor/src/train.py

4. Start prediction API:
python models/crime_predictor/src/predict.py

📝 Contribution & CI/CD

This project is designed to be production ready with GitHub Actions workflows (tests & model builds).
Contributions welcome 🌟

🛠️ Tech Stack

Core: Python 3.13, scikit-learn, joblib
Future: FastAPI, MLflow, Docker, GitHub Actions
Data: data.gouv.fr (police/gendarmerie 2016-2025)

📝 Author

Frédéric Tellier – Data Scientist
LinkedIn : https://www.linkedin.com/in/fr%C3%A9d%C3%A9ric-tellier-8a9170283/ | Portfolio : https://github.com/Dreipfelt/

Licence: MIT
=======
Author: Frédéric Tellier – Data Scientist
📜 License

MIT License
