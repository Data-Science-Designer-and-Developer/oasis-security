# 🛡️ Oasis Security — Crime Predictor

[![CI/CD](https://github.com/Data-Science-Designer-and-Developer/oasis-security/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/Data-Science-Designer-and-Developer/oasis-security/actions)
[![Docker](https://img.shields.io/badge/Docker-GHCR-blue)](https://ghcr.io/Data-Science-Designer-and-Developer)
[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

> **Modèle prédictif des taux de criminalité en France** (pour 100 000 habitants)  
> Pipeline ML complet : collecte → nettoyage → modélisation → API → dashboard  
> Source officielle : SSMSI / [data.gouv.fr](https://www.data.gouv.fr) — 2016-2023

---

## 📋 Sommaire

1. [Contexte & objectifs](#-contexte--objectifs)
2. [Structure du projet](#-structure-du-projet)
3. [Installation & lancement](#-installation--lancement)
4. [Pipeline de données](#-pipeline-de-données)
5. [Modélisation & résultats](#-modélisation--résultats)
6. [Dashboard Streamlit](#-dashboard-streamlit)
7. [API FastAPI](#-api-fastapi)
8. [Tests](#-tests)
9. [Docker & CI/CD](#-docker--cicd)
10. [Éthique & limites](#-éthique--limites)

---

## 🎯 Contexte & objectifs

Ce projet prédit les **taux de criminalité départementaux français** par catégorie d'infraction, à partir des données officielles de la police et de la gendarmerie nationales.

**Cas d'usage principal** : outil d'exploration statistique pour journalistes, chercheurs en sciences sociales et décideurs de politiques publiques.

**Objectifs techniques** :
- Construire un pipeline ML reproductible de bout en bout
- Comparer plusieurs algorithmes de régression avec tracking MLflow
- Déployer une API de prédiction (FastAPI) et un dashboard interactif (Streamlit)
- Appliquer les bonnes pratiques MLOps : versioning, tests, CI/CD, Docker

---

## 📁 Structure du projet

```
oasis-security/
├── .github/
│   └── workflows/          # CI/CD GitHub Actions
├── data/                   # Données nettoyées (.parquet)
├── docs/
│   └── crime_predictor/    # Documentation technique
├── images/                 # Visualisations & plots
├── models/
│   └── crime_predictor/
│       ├── src/
│       │   ├── train.py    # ← Pipeline d'entraînement (comparaison modèles)
│       │   └── predict.py  # ← API FastAPI
│       ├── models/
│       │   ├── crime_predictor.pkl   # Modèle sérialisé
│       │   └── metrics.json          # Métriques train/test
│       ├── mlruns/         # Expériences MLflow
│       └── tests/
│           └── test_model.py  # ← Tests unitaires
├── notebooks/              # Exploration & EDA
├── pipeline/               # Scripts d'automatisation
├── streamlit/              # Assets Streamlit complémentaires
├── app.py                  # ← Dashboard Streamlit principal
├── script_crimes_et_delits.py  # ← Collecte & nettoyage des données
├── Dockerfile              # Multi-stage build (train → production)
├── docker-compose.yml      # Stack complète (MLflow + Postgres + API)
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation & lancement

### 1. Cloner & installer

```bash
git clone https://github.com/Data-Science-Designer-and-Developer/oasis-security.git
cd oasis-security
python3.11 -m venv .venv
source .venv/bin/activate          # Windows : .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Télécharger et nettoyer les données

```bash
python script_crimes_et_delits.py
# → génère data/crimes_clean.parquet
```

### 3. Entraîner le modèle

```bash
python models/crime_predictor/src/train.py
# → compare 4 modèles, log dans MLflow, sauvegarde le meilleur
# → génère models/crime_predictor/models/crime_predictor.pkl
# → génère models/crime_predictor/models/metrics.json
```

### 4. Lancer le dashboard

```bash
streamlit run app.py
# → http://localhost:8501
```

### 5. Lancer l'API

```bash
uvicorn models.crime_predictor.src.predict:app --reload --port 8000
# → http://localhost:8000/docs
```

---

## 🔄 Pipeline de données

```
data.gouv.fr (SSMSI)
        ↓
script_crimes_et_delits.py
        ├── Téléchargement CSV (requests)
        ├── Normalisation colonnes (snake_case)
        ├── Suppression doublons
        ├── Cast types numériques
        ├── Suppression taux aberrants (< 0)
        ├── Feature engineering
        │   ├── taux_variation_annuelle (pct_change par dep × catégorie)
        │   └── annee_norm (normalisée [0, 1])
        └── Sauvegarde Parquet (Snappy)
                ↓
        data/crimes_clean.parquet
```

**Données brutes** : 8 colonnes, ~50 000 lignes  
**Après nettoyage** : 10 colonnes, ~49 000 lignes (< 2% de perte)

---

## 🤖 Modélisation & résultats

### Features utilisées

| Feature | Description |
|---|---|
| `annee` | Année (int) |
| `dep_encoded` | Département (LabelEncoded) |
| `cat_encoded` | Catégorie d'infraction (LabelEncoded) |
| `annee_norm` | Année normalisée [0, 1] |

**Cible** : `tauxpour100000hab` (taux d'infractions pour 100 000 habitants)  
**Split** : 80% train / 20% test — seed 42  
**Validation** : K-Fold cross-validation (k=5) sur le jeu d'entraînement

### Comparaison des modèles (jeu de test)

| Modèle | R² test | RMSE | MAE | CV R² (±std) |
|---|---|---|---|---|
| Ridge | 0.71 | 87.4 | 62.1 | 0.69 ± 0.03 |
| Random Forest | 0.89 | 54.2 | 38.7 | 0.87 ± 0.02 |
| Gradient Boosting | 0.88 | 56.1 | 40.2 | 0.86 ± 0.02 |
| **XGBoost** ✅ | **0.91** | **49.8** | **35.3** | **0.90 ± 0.01** |

> **Meilleur modèle : XGBoost** — R²=0.91 sur le jeu de test  
> Faible écart train/test → pas d'overfitting significatif  
> Faible variance cross-validation → robustesse confirmée

### Tracking MLflow

```bash
mlflow ui --backend-store-uri models/crime_predictor/mlruns
# → http://localhost:5000
```

---

## 📊 Dashboard Streamlit

5 pages interactives :

| Page | Contenu |
|---|---|
| Vue d'ensemble | KPIs, boxplot par catégorie, top 10 départements |
| Analyse départementale | Comparaison multi-dép., heatmap |
| Tendances temporelles | Évolution 2016-2023, indice base 100, variation annuelle |
| Prédiction ML | Simulateur interactif avec graphique historique |
| Éthique & Limites | Documentation des biais et limites d'usage |

---

## 🌐 API FastAPI

### Endpoints

| Méthode | Endpoint | Description |
|---|---|---|
| GET | `/health` | Statut API + métriques modèle |
| POST | `/predict` | Prédiction du taux |
| GET | `/docs` | Documentation Swagger interactive |

### Exemple de requête

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"annee": 2025, "dep_encoded": 5, "cat_encoded": 0, "annee_norm": 1.0}'
```

```json
{
  "taux_predit": 312.47,
  "unite": "infractions pour 100 000 habitants",
  "modele_utilise": "XGBoost",
  "r2_test": 0.91
}
```

---

## 🧪 Tests

```bash
# Lancer tous les tests
pytest models/crime_predictor/tests/ -v

# Avec couverture de code
pytest models/crime_predictor/tests/ -v --cov=models/crime_predictor/src --cov-report=term-missing
```

**Couverture des tests** :

| Classe | Tests |
|---|---|
| `TestData` | Intégrité du DataFrame (6 assertions) |
| `TestModel` | Forme, type, positivité, R², déterminisme (7 assertions) |
| `TestSerialization` | Sérialisation joblib, structure metrics.json (2 assertions) |

---

## 🐳 Docker & CI/CD

### Docker multi-stage

```bash
# Build (stage trainer → production)
docker build -t oasis-security:latest .

# Run l'API
docker run -p 8000:8000 oasis-security:latest
```

### Stack complète (MLflow + Postgres + API)

```bash
docker-compose up -d
# MLflow UI  → http://localhost:5000
# API        → http://localhost:8000/docs
```

### CI/CD GitHub Actions

Le workflow `.github/workflows/ci-cd.yml` déclenche à chaque push :
1. Lint (flake8)
2. Tests unitaires (pytest)
3. Build Docker
4. Push image sur GHCR

---

## ⚠️ Éthique & limites

> Ce modèle est un **outil d'exploration statistique**, non un système de décision opérationnel.

**Limites des données** :
- Ne couvre que les infractions *enregistrées* (chiffre noir estimé à 50-80%)
- Hétérogénéité des pratiques d'enregistrement entre services
- Pas de données infra-départementales

**Biais du modèle** :
- Reproduit les biais inhérents aux pratiques de signalement
- Corrélations ≠ causalité
- Non adapté aux chocs exogènes (COVID, crises économiques)

**Usages interdits** :
- Ciblage prédictif d'individus ou de zones géographiques
- Aide à la décision judiciaire ou pénale

**Conformité** : données agrégées anonymisées open data — aucune donnée personnelle.

---

## 📜 Licence

MIT — voir [LICENSE](LICENSE)

---

## 👤 Auteur

**Frédéric Tellier** — Data Scientist  
[LinkedIn](https://www.linkedin.com/in/fr%C3%A9d%C3%A9ric-tellier-8a9170283/) | [Portfolio](https://github.com/Dreipfelt/)

---

*Projet réalisé dans le cadre de la certification CDSD — 2025*
