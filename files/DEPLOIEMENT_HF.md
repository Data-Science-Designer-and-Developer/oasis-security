# 🚀 Guide de déploiement — Module Délinquance sur oasis-web

## Ce que tu vas faire
Ajouter une nouvelle page "Délinquance" à l'app Streamlit existante sur Hugging Face Spaces.

---

## Étape 1 — Exporter les données depuis le notebook

Dans Jupyter, exécute la dernière cellule du notebook `benchmark_delinquance_regional.ipynb`
(section "EXPORT POUR DÉPLOIEMENT HUGGING FACE").

Elle génère 3 fichiers dans un dossier `data/` local :
- `delinquance_region.csv`        (~5 Mo)
- `delinquance_previsions_2030.csv` (~2 Mo)
- `delinquance_benchmark.csv`     (~1 Mo)

---

## Étape 2 — Cloner le Space en local

```bash
# Installer git-lfs si pas déjà fait
git lfs install

# Cloner le repo HF Space
git clone https://huggingface.co/spaces/oasisorg/oasis-web
cd oasis-web
```

---

## Étape 3 — Copier les fichiers

```bash
# Les 3 fichiers CSV de données
cp /chemin/vers/ton/notebook/data/delinquance_region.csv        data/
cp /chemin/vers/ton/notebook/data/delinquance_previsions_2030.csv data/
cp /chemin/vers/ton/notebook/data/delinquance_benchmark.csv      data/

# La page Streamlit
cp /chemin/vers/10_Delinquance.py  src/pages/10_Delinquance.py

# Le requirements.txt mis à jour
cp /chemin/vers/requirements.txt   requirements.txt
```

---

## Étape 4 — Mettre à jour le Dockerfile

Ajoute ces 3 lignes dans le Dockerfile existant, après les autres COPY :

```dockerfile
COPY --chown=user data/delinquance_region.csv           ./data/
COPY --chown=user data/delinquance_previsions_2030.csv  ./data/
COPY --chown=user data/delinquance_benchmark.csv        ./data/
COPY --chown=user src/pages/10_Delinquance.py           ./src/pages/10_Delinquance.py
```

---

## Étape 5 — Pousser sur Hugging Face

```bash
git add data/delinquance_*.csv
git add src/pages/10_Delinquance.py
git add requirements.txt
git add Dockerfile

git commit -m "feat: add delinquance analysis page (historical + 2030 predictions)"
git push
```

HF rebuilde automatiquement le Docker (~3-5 min).

---

## Étape 6 — Vérifier

Ouvre https://huggingface.co/spaces/oasisorg/oasis-web
La page "10 Delinquance" apparaît dans la sidebar.

---

## Structure finale du repo

```
oasis-web/
├── data/
│   ├── delinquance_region.csv              ← NOUVEAU
│   ├── delinquance_previsions_2030.csv     ← NOUVEAU
│   ├── delinquance_benchmark.csv           ← NOUVEAU
│   └── ... (fichiers existants)
├── src/
│   ├── Home.py
│   └── pages/
│       ├── 0_Summary.py
│       ├── 1_Historical_Prices.py
│       ├── 2_Prediction_Prices.py
│       ├── ...
│       └── 10_Delinquance.py               ← NOUVEAU
├── Dockerfile                              ← MODIFIÉ
├── requirements.txt                        ← MODIFIÉ
└── README.md
```

---

## En cas de problème

### "Module prophet not found" au build
→ Normal : prophet prend ~2 min à installer. Attendre que le build se termine.

### "File not found: delinquance_region.csv"
→ Vérifier que DATA_DIR dans 10_Delinquance.py pointe bien vers `../../data/`
  depuis `src/pages/`. Ajuster si la structure du repo est différente.

### Build timeout
→ Les modèles ML (prophet, xgboost, lightgbm) alourdissent le build.
  Si le Space gratuit timeout, passer à un Space "CPU upgrade" ($0.05/h sur HF).

### La page ne s'affiche pas
→ Vérifier que le nom du fichier commence bien par un chiffre : `10_Delinquance.py`
  Streamlit utilise ce chiffre pour l'ordre dans la sidebar.
