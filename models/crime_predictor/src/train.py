"""
train.py — Pipeline d'entraînement du Crime Predictor
------------------------------------------------------
Compare plusieurs modèles, log les expériences dans MLflow,
sauvegarde le meilleur modèle (LightGBM) et exporte les métriques.

Usage :
    python models/crime_predictor/src/train.py
    python models/crime_predictor/src/train.py --data-path data/crimes_clean.parquet
"""

import argparse
import json
import logging
import os
import warnings
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import mlflow.lightgbm
import numpy as np
import pandas as pd
import yaml
from lightgbm import LGBMRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Chemins
# ─────────────────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).resolve().parents[3]
DEFAULT_DATA = ROOT / "data" / "crimes_clean.parquet"
MODEL_DIR    = Path(__file__).resolve().parents[1] / "artifacts"
CONFIG_PATH  = Path(__file__).resolve().parent / "config.yaml"
METRICS_PATH = MODEL_DIR / "metrics.json"
MODEL_PATH   = MODEL_DIR / "crime_predictor.pkl"
MLFLOW_URI   = os.getenv(
    "MLFLOW_TRACKING_URI",
    str(Path(__file__).resolve().parents[3] / "mlruns")
)

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
with open(CONFIG_PATH, "r") as f:
    CONFIG = yaml.safe_load(f)

LGB_PARAMS = CONFIG["model"]

# ─────────────────────────────────────────────────────────────────────────────
# Modèles candidats — benchmark complet
# ─────────────────────────────────────────────────────────────────────────────
CANDIDATES = {
    "Ridge": Pipeline([
        ("scaler", StandardScaler()),
        ("model",  Ridge(alpha=1.0)),
    ]),
    "RandomForest": RandomForestRegressor(
        n_estimators=200, max_depth=8, random_state=42, n_jobs=-1
    ),
    "GradientBoosting": GradientBoostingRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42
    ),
    "XGBoost": XGBRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        random_state=42, verbosity=0,
    ),
    # Champion — hyperparamètres issus de config.yaml
    "LightGBM": LGBMRegressor(**LGB_PARAMS),
}


# ─────────────────────────────────────────────────────────────────────────────
# Chargement données
# ─────────────────────────────────────────────────────────────────────────────
def load_data(path: Path) -> pd.DataFrame:
    """
    Charge le fichier Parquet préprocessé.
    Si le fichier est absent, lève une erreur explicite avec les instructions
    pour générer le fichier depuis la source data.gouv.fr.
    """
    if path.exists():
        df = pd.read_parquet(path)
        logger.info("Données chargées : %s — %d lignes", path, len(df))
        return df

    raise FileNotFoundError(
        f"\n\n[ERREUR] Fichier de données introuvable : {path}\n"
        "Pour générer ce fichier, lancez d'abord le pipeline de prétraitement :\n"
        "    python pipeline/preprocess.py\n\n"
        "Ou téléchargez directement la source :\n"
        f"    {CONFIG['data']['url']}\n"
        "puis relancez le script avec :\n"
        "    python models/crime_predictor/src/train.py --data-path <chemin_vers_csv>\n"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Feature engineering
# ─────────────────────────────────────────────────────────────────────────────
def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Features :
        - annee           (int brut)
        - annee_norm      (normalisé [0, 1])
        - year_sin/cos    (encodage cyclique)
        - dep_encoded     (label encoding département)
        - cat_encoded     (label encoding catégorie)

    Cible :
        - tauxpour100000hab  (ou taux_100k selon la colonne disponible)
    """
    df = df.copy()

    # Détection colonne cible
    target_col = "taux_100k" if "taux_100k" in df.columns else "tauxpour100000hab"
    if target_col not in df.columns:
        raise ValueError(
            f"Colonne cible introuvable. Colonnes disponibles : {list(df.columns)}"
        )

    le_dep = LabelEncoder()
    le_cat = LabelEncoder()

    df["dep_encoded"] = le_dep.fit_transform(df["dep"].astype(str))
    df["cat_encoded"] = le_cat.fit_transform(df["indicateur"].astype(str))
    df["annee_norm"]  = (df["annee"] - df["annee"].min()) / max(
        df["annee"].max() - df["annee"].min(), 1
    )
    df["year_sin"] = np.sin(2 * np.pi * df["annee"] / 10)
    df["year_cos"] = np.cos(2 * np.pi * df["annee"] / 10)

    FEATURES = ["annee", "dep_encoded", "cat_encoded", "annee_norm",
                "year_sin", "year_cos"]
    X = df[FEATURES]
    y = df[target_col]
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# Métriques
# ─────────────────────────────────────────────────────────────────────────────
def evaluate(model, X: pd.DataFrame, y: pd.Series) -> dict:
    """Calcule R², RMSE, MAE."""
    preds = model.predict(X)
    return {
        "r2_test":   round(float(r2_score(y, preds)),                     4),
        "rmse_test": round(float(np.sqrt(mean_squared_error(y, preds))),  4),
        "mae_test":  round(float(mean_absolute_error(y, preds)),           4),
    }


def cross_validate_model(model, X: pd.DataFrame, y: pd.Series, cv: int = 3) -> dict:
    """
    Cross-validation temporelle (TimeSeriesSplit) — respecte l'ordre chronologique
    et évite toute fuite de données futures vers le passé.
    """
    tscv = TimeSeriesSplit(n_splits=cv)
    r2_scores = cross_val_score(model, X, y, cv=tscv, scoring="r2")
    return {
        "cv_r2_mean": round(float(r2_scores.mean()), 4),
        "cv_r2_std":  round(float(r2_scores.std()),  4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main(data_path: Path) -> None:
    logger.info("=== Pipeline d'entraînement — Oasis Security ===")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Données
    df = load_data(data_path)
    X, y = build_features(df)

    # Split temporel : 80% train / 20% test — sans shuffle (données temporelles)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )
    logger.info("Train : %d | Test : %d", len(X_train), len(X_test))

    # 2. MLflow
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(CONFIG["mlflow"]["experiment_name"])

    best_model  = None
    best_name   = ""
    best_r2     = -np.inf
    all_results = {}

    # 3. Benchmark des modèles candidats
    logger.info("Benchmark des modèles candidats...")
    for name, candidate in CANDIDATES.items():
        with mlflow.start_run(run_name=name):

            candidate.fit(X_train, y_train)

            metrics_test  = evaluate(candidate, X_test, y_test)
            metrics_train = evaluate(candidate, X_train, y_train)
            cv_metrics    = cross_validate_model(candidate, X_train, y_train, cv=3)

            all_metrics = {
                **metrics_test,
                **cv_metrics,
                "r2_train": metrics_train["r2_test"],
            }
            all_results[name] = all_metrics

            mlflow.log_params({"model": name})
            mlflow.log_metrics(all_metrics)

            # Log artifact LightGBM avec le module dédié pour meilleure traçabilité
            if name == "LightGBM":
                mlflow.lightgbm.log_model(
                    candidate,
                    artifact_path="model",
                    registered_model_name=CONFIG["mlflow"]["registered_model_name"],
                )
            else:
                mlflow.sklearn.log_model(candidate, artifact_path="model")

            logger.info(
                "%-20s | R²_test=%.4f | RMSE=%.2f | MAE=%.2f | CV_R²=%.4f±%.4f",
                name,
                metrics_test["r2_test"],
                metrics_test["rmse_test"],
                metrics_test["mae_test"],
                cv_metrics["cv_r2_mean"],
                cv_metrics["cv_r2_std"],
            )

            if metrics_test["r2_test"] > best_r2:
                best_r2    = metrics_test["r2_test"]
                best_model = candidate
                best_name  = name

    # 4. Sauvegarde du champion
    logger.info("Champion : %s (R²_test=%.4f)", best_name, best_r2)
    joblib.dump(best_model, MODEL_PATH)
    logger.info("Modèle sauvegardé : %s", MODEL_PATH)

    # 5. Export métriques
    best_metrics = {
        "best_model": best_name,
        **all_results[best_name],
        "all_models": all_results,
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(best_metrics, f, indent=2, ensure_ascii=False)
    logger.info("Métriques exportées : %s", METRICS_PATH)

    # 6. Rapport final
    print("\n" + "=" * 70)
    print(f"{'Modèle':<22} {'R²_test':>8} {'RMSE':>8} {'MAE':>8} {'CV_R²':>12}")
    print("-" * 70)
    for name, m in all_results.items():
        marker = " ← CHAMPION" if name == best_name else ""
        print(
            f"{name:<22} {m['r2_test']:>8.4f} {m['rmse_test']:>8.2f} "
            f"{m['mae_test']:>8.2f} {m['cv_r2_mean']:>6.4f}±{m['cv_r2_std']:.4f}"
            f"{marker}"
        )
    print("=" * 70)
    logger.info("=== Entraînement terminé ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Crime Predictor — Oasis Security")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA,
        help="Chemin vers crimes_clean.parquet (généré par pipeline/preprocess.py)",
    )
    args = parser.parse_args()
    main(args.data_path)
