"""
app.py — Oasis Security | Crime Predictor Dashboard
----------------------------------------------------
Application Streamlit de visualisation et prédiction des taux
de criminalité en France (pour 100 000 habitants).

Usage :
    streamlit run app.py
"""

import os
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Oasis Security — Crime Predictor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  [data-testid="stMetricValue"] { font-size: 1.8rem; font-weight: 700; }
  .block-container { padding-top: 1.5rem; }
  h1 { color: #1a2744; }
  h2, h3 { color: #2c3e6b; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Données
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def load_data() -> pd.DataFrame:
    """
    Charge data/crimes_clean.parquet si disponible.
    Sinon, génère des données synthétiques réalistes (SSMSI 2016-2023)
    pour permettre une démonstration complète sans pré-requis.
    """
    path = os.path.join("data", "crimes_clean.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)

    # ── Données synthétiques (proche des vraies statistiques SSMSI) ──────────
    np.random.seed(42)
    ANNEES = list(range(2016, 2024))
    DEPS = {
        "75": "Paris", "13": "Bouches-du-Rhône", "69": "Rhône",
        "59": "Nord", "33": "Gironde", "31": "Haute-Garonne",
        "06": "Alpes-Mar.", "44": "Loire-Atl.", "67": "Bas-Rhin",
        "76": "Seine-Mar.", "92": "Hauts-de-Seine", "93": "Seine-St-Denis",
        "94": "Val-de-Marne", "78": "Yvelines", "91": "Essonne",
        "38": "Isère", "34": "Hérault", "57": "Moselle",
        "14": "Calvados", "37": "Indre-et-Loire",
    }
    CATEGORIES = {
        "Cambriolages de logement":        {"base": 280, "trend": -0.04},
        "Vols avec violence":               {"base": 185, "trend":  0.01},
        "Vols sans violence contre pers.":  {"base": 950, "trend": -0.02},
        "Coups et blessures volontaires":   {"base": 520, "trend":  0.03},
        "Escroqueries":                     {"base": 380, "trend":  0.05},
    }

    rows = []
    for dep_code, dep_nom in DEPS.items():
        coef = np.random.uniform(0.5, 1.9)
        for cat, params in CATEGORIES.items():
            for i, annee in enumerate(ANNEES):
                taux = (
                    params["base"]
                    * coef
                    * (1 + params["trend"]) ** i
                    * np.random.normal(1, 0.04)
                )
                rows.append({
                    "annee":              annee,
                    "dep":                dep_code,
                    "dep_nom":            dep_nom,
                    "indicateur":         cat,
                    "tauxpour100000hab":  round(max(taux, 0), 2),
                })

    return pd.DataFrame(rows)


@st.cache_resource
def load_model():
    """Charge le modèle .pkl si disponible."""
    import joblib
    path = os.path.join("models", "crime_predictor", "models", "crime_predictor.pkl")
    if os.path.exists(path):
        try:
            return joblib.load(path)
        except Exception:
            return None
    return None


df    = load_data()
model = load_model()

ANNEES      = sorted(df["annee"].unique())
CATEGORIES  = sorted(df["indicateur"].unique())
DEPS        = sorted(df["dep"].unique())
DEP_LABELS  = df[["dep", "dep_nom"]].drop_duplicates().set_index("dep")["dep_nom"].to_dict()

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🛡️ Oasis Security")
    st.caption("Crime Predictor — France 2016-2023")
    st.markdown("---")

    page = st.selectbox(
        "Navigation",
        ["📊 Vue d'ensemble",
         "🗺️ Analyse départementale",
         "📈 Tendances temporelles",
         "🤖 Prédiction ML",
         "⚠️ Éthique & Limites"],
    )

    st.markdown("---")
    annee_ref = st.slider("Année de référence", int(min(ANNEES)), int(max(ANNEES)), int(max(ANNEES)))
    cats_sel  = st.multiselect("Catégories", CATEGORIES, default=CATEGORIES[:3])
    if not cats_sel:
        cats_sel = CATEGORIES[:1]

    st.markdown("---")
    st.markdown(
        "📂 **Source** : [data.gouv.fr](https://www.data.gouv.fr) — SSMSI  \n"
        "👤 **Auteur** : Frédéric Tellier  \n"
        "[![LinkedIn](https://img.shields.io/badge/LinkedIn-profil-blue)]"
        "(https://www.linkedin.com/in/fr%C3%A9d%C3%A9ric-tellier-8a9170283/)"
    )

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1 — Vue d'ensemble
# ─────────────────────────────────────────────────────────────────────────────
if page == "📊 Vue d'ensemble":

    st.title("📊 Vue d'ensemble")
    st.markdown(
        f"Analyse des taux de criminalité pour **{annee_ref}** "
        f"— {len(DEPS)} départements · {len(CATEGORIES)} catégories"
    )

    df_ref = df[df["annee"] == annee_ref]

    # ── KPIs ─────────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Taux moyen national",  f"{df_ref['tauxpour100000hab'].mean():.0f} / 100k")
    c2.metric("Taux médian",          f"{df_ref['tauxpour100000hab'].median():.0f} / 100k")
    c3.metric("Département le + touché",
              df_ref.groupby("dep")["tauxpour100000hab"].mean().idxmax())
    c4.metric("Catégorie la + fréquente",
              df_ref.groupby("indicateur")["tauxpour100000hab"].mean().idxmax())

    st.markdown("---")

    # ── Boxplot par catégorie ─────────────────────────────────────────────────
    st.subheader("Distribution des taux par catégorie d'infraction")
    df_box = df_ref[df_ref["indicateur"].isin(cats_sel)]
    fig_box = px.box(
        df_box, x="indicateur", y="tauxpour100000hab",
        color="indicateur",
        labels={"tauxpour100000hab": "Taux / 100 000 hab", "indicateur": ""},
        color_discrete_sequence=px.colors.qualitative.Bold,
        height=420,
    )
    fig_box.update_layout(showlegend=False, xaxis_tickangle=-20)
    st.plotly_chart(fig_box, use_container_width=True)

    # ── Barplot top 10 départements ───────────────────────────────────────────
    st.subheader(f"Top 10 départements — taux moyen toutes catégories ({annee_ref})")
    top10 = (
        df_ref.groupby(["dep", "dep_nom"])["tauxpour100000hab"]
        .mean().reset_index()
        .sort_values("tauxpour100000hab", ascending=False)
        .head(10)
    )
    fig_bar = px.bar(
        top10, x="dep_nom", y="tauxpour100000hab",
        color="tauxpour100000hab", color_continuous_scale="Reds",
        labels={"tauxpour100000hab": "Taux moyen / 100k", "dep_nom": "Département"},
        height=380,
    )
    fig_bar.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig_bar, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2 — Analyse départementale
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🗺️ Analyse départementale":

    st.title("🗺️ Analyse départementale")

    deps_comp = st.multiselect(
        "Départements à comparer",
        options=DEPS,
        format_func=lambda d: f"{d} — {DEP_LABELS.get(d, d)}",
        default=DEPS[:5],
    )
    cat_comp = st.selectbox("Catégorie d'infraction", CATEGORIES)

    df_comp = df[(df["dep"].isin(deps_comp)) & (df["indicateur"] == cat_comp)]

    # ── Évolution par département ─────────────────────────────────────────────
    fig_line = px.line(
        df_comp, x="annee", y="tauxpour100000hab", color="dep_nom",
        markers=True,
        title=f"{cat_comp} — évolution par département",
        labels={"tauxpour100000hab": "Taux / 100k", "annee": "Année", "dep_nom": "Département"},
        height=440,
    )
    st.plotly_chart(fig_line, use_container_width=True)

    # ── Heatmap (dép × catégorie) ─────────────────────────────────────────────
    st.subheader(f"Heatmap — tous indicateurs ({annee_ref})")
    pivot = (
        df[df["annee"] == annee_ref]
        .groupby(["dep_nom", "indicateur"])["tauxpour100000hab"]
        .mean()
        .unstack("indicateur")
        .fillna(0)
    )
    fig_heat = px.imshow(
        pivot, aspect="auto",
        color_continuous_scale="RdYlGn_r",
        title=f"Taux / 100k hab — {annee_ref}",
        labels={"color": "Taux / 100k"},
        height=520,
    )
    st.plotly_chart(fig_heat, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3 — Tendances temporelles
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📈 Tendances temporelles":

    st.title("📈 Tendances temporelles")
    st.markdown("Évolution des taux moyens nationaux de 2016 à 2023.")

    # ── Tendances absolues ────────────────────────────────────────────────────
    df_nat = (
        df[df["indicateur"].isin(cats_sel)]
        .groupby(["annee", "indicateur"])["tauxpour100000hab"]
        .mean().reset_index()
    )
    fig_abs = px.line(
        df_nat, x="annee", y="tauxpour100000hab", color="indicateur",
        markers=True,
        title="Évolution nationale (taux moyen / 100k)",
        labels={"tauxpour100000hab": "Taux / 100k", "annee": "Année", "indicateur": "Catégorie"},
        height=440,
    )
    st.plotly_chart(fig_abs, use_container_width=True)

    # ── Indice base 100 = 2016 ────────────────────────────────────────────────
    st.subheader("Indice base 100 (référence : 2016)")
    pivot_nat = df_nat.pivot(index="annee", columns="indicateur", values="tauxpour100000hab")
    base = pivot_nat.loc[2016]
    base100 = (pivot_nat / base * 100).reset_index().melt(
        id_vars="annee", var_name="indicateur", value_name="indice"
    )
    fig_b100 = px.line(
        base100, x="annee", y="indice", color="indicateur",
        markers=True,
        labels={"indice": "Indice (base 100)", "annee": "Année"},
        height=420,
    )
    fig_b100.add_hline(y=100, line_dash="dash", line_color="#999",
                       annotation_text="Base 2016", annotation_position="right")
    st.plotly_chart(fig_b100, use_container_width=True)

    # ── Variation annuelle moyenne ────────────────────────────────────────────
    st.subheader("Variation annuelle moyenne par catégorie")
    var_moy = (
        df_nat.sort_values("annee")
        .groupby("indicateur")["tauxpour100000hab"]
        .apply(lambda s: s.pct_change().mean() * 100)
        .reset_index()
        .rename(columns={"tauxpour100000hab": "variation_moy_pct"})
        .sort_values("variation_moy_pct")
    )
    colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in var_moy["variation_moy_pct"]]
    fig_var = go.Figure(go.Bar(
        x=var_moy["variation_moy_pct"].round(2),
        y=var_moy["indicateur"],
        orientation="h",
        marker_color=colors,
        text=var_moy["variation_moy_pct"].apply(lambda v: f"{v:+.1f}%"),
        textposition="outside",
    ))
    fig_var.update_layout(
        title="Variation annuelle moyenne 2016-2023 (%)",
        xaxis_title="Variation (%)", yaxis_title="",
        height=350,
    )
    st.plotly_chart(fig_var, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 4 — Prédiction ML
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🤖 Prédiction ML":

    st.title("🤖 Prédiction ML")

    if model is not None:
        st.success("✅ Modèle chargé — `models/crime_predictor/models/crime_predictor.pkl`")
    else:
        st.info(
            "ℹ️ Modèle non trouvé localement — prédiction par extrapolation linéaire. "
            "Pour charger le modèle entraîné : `python models/crime_predictor/src/train.py`"
        )

    st.markdown("### Simulateur de prédiction")
    st.markdown(
        "Sélectionnez un département et une catégorie pour estimer "
        "le taux de criminalité attendu sur une année future."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        dep_pred = st.selectbox(
            "Département",
            DEPS,
            format_func=lambda d: f"{d} — {DEP_LABELS.get(d, d)}",
        )
    with col2:
        cat_pred = st.selectbox("Catégorie d'infraction", CATEGORIES, key="pred_cat")
    with col3:
        annee_pred = st.selectbox("Année à prédire", list(range(2024, 2027)))

    # Données historiques du département/catégorie sélectionné
    hist = df[(df["dep"] == dep_pred) & (df["indicateur"] == cat_pred)].sort_values("annee")

    # Prédiction : modèle ML si dispo, sinon régression linéaire simple
    if model is not None and len(hist) > 0:
        X_pred = np.array([[annee_pred]])
        pred_val = float(model.predict(X_pred)[0])
        method = "modèle ML entraîné (scikit-learn)"
    elif len(hist) >= 2:
        x = hist["annee"].values
        y = hist["tauxpour100000hab"].values
        coeffs = np.polyfit(x, y, 1)
        pred_val = float(np.polyval(coeffs, annee_pred))
        pred_val = max(pred_val, 0)
        method = "extrapolation linéaire (fallback)"
    else:
        pred_val = hist["tauxpour100000hab"].mean()
        method = "moyenne historique"

    # Affichage résultat
    st.markdown("---")
    rc1, rc2 = st.columns([1, 2])
    with rc1:
        st.metric(
            label=f"Taux prédit — {annee_pred}",
            value=f"{pred_val:.1f} / 100k",
            delta=f"{pred_val - hist.iloc[-1]['tauxpour100000hab']:.1f} vs {int(hist.iloc[-1]['annee'])}",
        )
        st.caption(f"Méthode : {method}")

    with rc2:
        # Graphique historique + prédiction
        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(
            x=hist["annee"], y=hist["tauxpour100000hab"],
            mode="lines+markers", name="Historique",
            line=dict(color="#3b5bdb", width=2),
            marker=dict(size=7),
        ))
        fig_pred.add_trace(go.Scatter(
            x=[int(hist.iloc[-1]["annee"]), annee_pred],
            y=[hist.iloc[-1]["tauxpour100000hab"], pred_val],
            mode="lines+markers", name="Prédiction",
            line=dict(color="#e74c3c", width=2, dash="dash"),
            marker=dict(size=10, symbol="star"),
        ))
        fig_pred.update_layout(
            title=f"{cat_pred} — Dép. {dep_pred} ({DEP_LABELS.get(dep_pred, '')})",
            xaxis_title="Année",
            yaxis_title="Taux / 100k hab",
            height=340,
            legend=dict(orientation="h", y=-0.2),
        )
        st.plotly_chart(fig_pred, use_container_width=True)

    # Métriques du modèle si disponibles
    metrics_path = os.path.join("models", "crime_predictor", "models", "metrics.json")
    if os.path.exists(metrics_path):
        import json
        with open(metrics_path) as f:
            metrics = json.load(f)
        st.markdown("### Performances du modèle (jeu de test)")
        m1, m2, m3 = st.columns(3)
        m1.metric("R²",    f"{metrics.get('r2_test', 'N/A'):.3f}")
        m2.metric("RMSE",  f"{metrics.get('rmse_test', 'N/A'):.2f}")
        m3.metric("MAE",   f"{metrics.get('mae_test', 'N/A'):.2f}")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 5 — Éthique & Limites
# ─────────────────────────────────────────────────────────────────────────────
elif page == "⚠️ Éthique & Limites":

    st.title("⚠️ Éthique & Limites du modèle")

    st.warning(
        "**Ce modèle est un outil d'aide à l'analyse statistique, "
        "non un système de décision opérationnel.**"
    )

    st.markdown("""
## 1. Limites des données

- **Chiffre noir de la criminalité** : les données SSMSI ne reflètent que les infractions
  *enregistrées* par la police et la gendarmerie. Les infractions non déclarées
  (estimées à 50-80 % selon les catégories) sont absentes.
- **Hétérogénéité des pratiques d'enregistrement** : les taux varient selon
  les territoires non seulement pour des raisons criminologiques, mais aussi
  du fait de différences d'enregistrement entre services.
- **Périmètre géographique** : les données couvrent la France métropolitaine,
  hors certains territoires d'outre-mer.

## 2. Biais du modèle

- **Biais de représentation** : entraîné sur des infractions signalées,
  le modèle reproduit les biais inhérents aux pratiques de signalement.
  Les populations qui signalent moins (précarité, méfiance institutionnelle)
  sont sous-représentées.
- **Corrélations ≠ causalité** : une corrélation entre variables socio-économiques
  et taux de criminalité n'implique aucune causalité. Toute interprétation
  causaliste est abusive.
- **Stationnarité** : le modèle suppose une certaine stationnarité des tendances.
  Des chocs exogènes (COVID-2020, crises économiques) ne sont pas modélisés.

## 3. Risques d'usage

| Usage | Statut |
|---|---|
| Exploration statistique et journalisme de données | ✅ Approprié |
| Aide à l'allocation de ressources de sécurité publique | ⚠️ Nécessite validation experte |
| Ciblage prédictif d'individus ou de zones | 🚫 Hors périmètre — risque discriminatoire |
| Décision judiciaire ou pénale | 🚫 Interdit |

## 4. Conformité RGPD

Les données utilisées sont des **statistiques agrégées anonymisées**
publiées en open data par le gouvernement français. Aucune donnée personnelle
n'est traitée dans ce projet.

## 5. Recommandations

- Ce modèle doit être interprété par des experts en criminologie ou en politiques publiques.
- Toute décision opérationnelle doit combiner ce modèle avec d'autres sources
  de connaissance terrain.
- Les résultats doivent être présentés avec leurs intervalles de confiance.
""")

    st.info(
        "📚 Pour aller plus loin : "
        "[Observatoire national de la délinquance (ONDRP)](https://www.ihesi.interieur.gouv.fr/) — "
        "[Rapport annuel SSMSI](https://www.interieur.gouv.fr/Interstats)"
    )
