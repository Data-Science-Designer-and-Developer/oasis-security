"""
script_crimes_et_delits.py
--------------------------
Téléchargement, nettoyage et feature engineering des données
de criminalité françaises (data.gouv.fr / SSMSI).

Source officielle :
    Service Statistique Ministériel de la Sécurité Intérieure (SSMSI)
    Crimes et délits enregistrés par la police et la gendarmerie — 2016-2023

Usage :
    python script_crimes_et_delits.py

Sortie :
    data/crimes_clean.parquet
"""

import logging
import os
import sys
from io import StringIO

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
DATA_URL = (
    "https://www.data.gouv.fr/fr/datasets/r/"
    "cf9b52c8-71d3-432f-863e-4a7e9b2d9bcf"
)
OUTPUT_DIR = "data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "crimes_clean.parquet")
TIMEOUT = 30


# ---------------------------------------------------------------------------
# Fonctions
# ---------------------------------------------------------------------------

def download_data(url: str) -> pd.DataFrame:
    """Télécharge le CSV officiel depuis data.gouv.fr."""
    logger.info("Téléchargement : %s", url)
    try:
        resp = requests.get(url, timeout=TIMEOUT)
        resp.raise_for_status()
    except requests.exceptions.RequestException as exc:
        logger.error("Échec du téléchargement : %s", exc)
        sys.exit(1)

    df = pd.read_csv(StringIO(resp.text), sep=";", encoding="utf-8")
    logger.info("Données brutes : %d lignes × %d colonnes", *df.shape)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoyage du DataFrame brut :
      1. Normalisation des noms de colonnes (snake_case)
      2. Suppression des doublons
      3. Cast des types numériques
      4. Suppression des valeurs aberrantes (taux < 0)
      5. Imputation / suppression des NaN sur la variable cible
    """
    logger.info("Nettoyage des données...")

    # 1. Noms de colonnes normalisés
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(r"[\s./]", "_", regex=True)
        .str.replace(r"[àáâ]", "a", regex=True)
        .str.replace(r"[éèê]", "e", regex=True)
        .str.replace(r"[îï]", "i", regex=True)
        .str.replace(r"[ôö]", "o", regex=True)
        .str.replace(r"[ùûü]", "u", regex=True)
    )

    # 2. Doublons
    before = len(df)
    df = df.drop_duplicates()
    logger.info("Doublons supprimés : %d", before - len(df))

    # 3. Cast numériques
    for col in ["valeur", "tauxpour100000hab", "pop"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 4. Taux négatifs = aberrant
    if "tauxpour100000hab" in df.columns:
        mask_neg = df["tauxpour100000hab"] < 0
        if mask_neg.sum() > 0:
            logger.warning("Taux négatifs retirés : %d lignes", mask_neg.sum())
        df = df[~mask_neg]

    # 5. NaN sur la cible
    df = df.dropna(subset=["tauxpour100000hab"])

    # 6. Année en int
    if "annee" in df.columns:
        df["annee"] = df["annee"].astype(int)

    logger.info("Après nettoyage : %d lignes × %d colonnes", *df.shape)
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering :
      - taux_variation_annuelle : variation % d'une année à l'autre
        par (département, catégorie d'infraction)
      - annee_norm : année normalisée [0, 1] pour les modèles linéaires
    """
    logger.info("Feature engineering...")

    if {"tauxpour100000hab", "annee", "dep", "indicateur"}.issubset(df.columns):
        df = df.sort_values(["dep", "indicateur", "annee"])
        df["taux_variation_annuelle"] = (
            df.groupby(["dep", "indicateur"])["tauxpour100000hab"]
            .pct_change()
            .round(4)
        )

    if "annee" in df.columns:
        annee_min = df["annee"].min()
        annee_max = df["annee"].max()
        df["annee_norm"] = (df["annee"] - annee_min) / max(annee_max - annee_min, 1)

    return df


def save_data(df: pd.DataFrame, path: str) -> None:
    """Sauvegarde en Parquet (compression Snappy par défaut)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False, engine="pyarrow")
    size_kb = os.path.getsize(path) / 1024
    logger.info("Sauvegardé : %s (%.1f Ko)", path, size_kb)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logger.info("=== Pipeline données — Oasis Security ===")
    df_raw = download_data(DATA_URL)
    df_clean = clean_data(df_raw)
    df_features = add_features(df_clean)
    save_data(df_features, OUTPUT_FILE)
    logger.info("=== Pipeline terminé avec succès ===")


if __name__ == "__main__":
    main()
