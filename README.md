---
title: Oasis Security
emoji: 🛡️
colorFrom: blue
colorTo: gray
sdk: streamlit
sdk_version: 1.45.0
python_version: '3.10'
app_file: app.py
pinned: false
---

# OASIS Security — Crime & Delinquency Analysis in France

An interactive dashboard for the analysis and forecasting of recorded crime and
delinquency in France, based on official data from the Police Nationale and
Gendarmerie Nationale (2016–2025), with projections through to 2030.

---

## 📊 Data

| Property | Details |
|---|---|
| **Source** | [data.gouv.fr](https://www.data.gouv.fr) — Base statistique régionale |
| **Publisher** | Police Nationale & Gendarmerie Nationale |
| **Granularity** | Regional (INSEE codes 2025) |
| **Period** | 2016–2025 |
| **Format** | CSV (semicolon-delimited) |

---

## 🤖 Modelling & Forecasting

- **Primary model:** Holt-Winters Exponential Smoothing (additive, damped trend)
  via `statsmodels`
- **Fallback model:** 2nd-degree polynomial regression (`numpy`)
- **Train/test split:** Last 2 years held out for evaluation
- **Forecast horizon:** Up to 2030
- **Evaluation metrics:** MAE, RMSE

---

## 🛠️ Technical Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| Visualisation | Plotly Express & Graph Objects |
| Data processing | Pandas, NumPy |
| Forecasting | Statsmodels, Scikit-learn |
| Deployment | Hugging Face Spaces |
| Containerisation | Docker (multi-stage build) |

---

## 🗂️ Features

- **KPI dashboard** — total recorded offences, year-on-year delta, crime type count
- **Time series** — historical trends per crime category with forecast overlay
- **Choropleth map** — regional distribution by year and crime type
- **Bar chart ranking** — top regions by volume of offences
- **Donut chart** — breakdown by crime category
- **Raw data export** — filtered CSV download

---

## ⚖️ Ethics & Data Privacy

The data used in this project is publicly available, aggregated at regional level,
and published by French government authorities. No individual-level data is
processed or displayed, and no re-identification of persons is possible from
the aggregated figures.

This dashboard is intended solely for informational and analytical purposes.
Forecasts are indicative and should not be used as the basis for operational
or policy decisions without further validation. The analysis does not carry
any discriminatory intent with respect to geographical areas or populations.

All data processing complies with the principles of the GDPR (Regulation (EU)
2016/679), in particular the principles of data minimisation and purpose limitation.

---

## 📁 Repository Structure

```
oasis-security/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── models/
│   └── crime_predictor/
│       ├── Dockerfile      # Multi-stage build (train → serve)
│       ├── train.py        # Model training pipeline
│       ├── predict.py      # FastAPI inference endpoint
│       ├── model.py        # Model class definition
│       └── config.yaml     # Hyperparameters & settings
└── README.md
```

---

## 🔗 Links

- 🚀 **Live app:** [Hugging Face Space](https://huggingface.co/spaces/Dreipfelt/oasis-security)
- 📦 **Source code:** [GitHub Repository](https://github.com/Data-Science-Designer-and-Developer/oasis-security)
- 📂 **Dataset:** [data.gouv.fr](https://www.data.gouv.fr/fr/datasets/bases-statistiques-communale-departementale-et-regionale-de-la-delinquance-enregistree-par-la-police-et-la-gendarmerie-nationales/)

---

*CDSD Certification Project — Data Science Designer & Developer (RNCP35288)*
