# 🚨 OASIS Security Module

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://oasis-security.streamlit.app)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Module complémentaire du projet [OASIS](https://github.com/nclsprsnw/oasis) pour l'analyse des données de sécurité publique en France.

![Dashboard Screenshot](docs/screenshots/dashboard.png)

## 📊 Fonctionnalités

- 📈 **Évolution temporelle** : Visualisez l'évolution des infractions sur plusieurs années
- 📊 **Analyse comparative** : Comparez plusieurs types d'infractions simultanément
- 📋 **Statistiques détaillées** : Accédez aux métriques clés (évolution, extremums, moyennes)
- 🗺️ **Analyse départementale** : Identifiez les départements les plus touchés
- 🎛️ **Filtres interactifs** : Personnalisez votre analyse avec des filtres dynamiques

## 🚀 Installation

### Prérequis

- Python 3.9 ou supérieur
- pip (gestionnaire de paquets Python)

### Installation locale

```bash
# 1. Cloner le repository
git clone https://github.com/Dreipfelt/oasis-security.git
cd oasis-security

# 2. Créer un environnement virtuel (recommandé)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Télécharger les données (voir section Données)

# 5. Lancer l'application
streamlit run app.py

L'application sera accessible à l'adresse : http://localhost:8501
📁 Données
Source

Les données proviennent du Ministère de l'Intérieur via data.gouv.fr.
Téléchargement

    Rendez-vous sur data.gouv.fr
    Recherchez "statistiques criminalité départements" ou "séries chronologiques sécurité"
    Téléchargez le fichier CSV
    Renommez-le en serieschrono-datagouv.csv
    Placez-le dans le dossier data/

Format attendu

Le fichier CSV doit contenir les colonnes suivantes :
Colonne	Description	Exemple
Unite_temps	Année	2023
Zone_geographique	Département	75-Paris
Valeurs	Nombre de cas	12345
Indicateur	Type d'infraction	Vols avec violence
Code_dep	Code département (optionnel)	75
🔗 Intégration au Projet OASIS

Ce module est conçu pour s'intégrer facilement au projet OASIS.
Option 1 : Comme page Streamlit

    Copiez app.py dans le dossier pages/ du projet OASIS
    Renommez-le en 5_🚨_Securite.py
    Adaptez les chemins de données si nécessaire

Option 2 : Comme module séparé

Ajoutez un lien vers ce module dans l'interface OASIS principale.
📂 Structure du Projet

text

oasis-security/
├── app.py                 # Application principale
├── requirements.txt       # Dépendances Python
├── README.md             # Documentation
├── LICENSE               # Licence MIT
├── .gitignore           # Fichiers à ignorer
├── .streamlit/
│   └── config.toml      # Configuration Streamlit
├── data/
│   ├── .gitkeep
│   └── README.md        # Instructions données
└── docs/
    ├── screenshots/
    │   └── dashboard.png
    └── integration_oasis.md

🛠️ Technologies Utilisées

    Streamlit - Framework web pour applications data
    Plotly - Bibliothèque de visualisation interactive
    Pandas - Manipulation de données
    NumPy - Calculs numériques

👥 Contribution

Les contributions sont les bienvenues !

    Forkez le projet
    Créez votre branche (git checkout -b feature/AmazingFeature)
    Committez vos changements (git commit -m 'Add: Amazing Feature')
    Pushez sur la branche (git push origin feature/AmazingFeature)
    Ouvrez une Pull Request

📜 Licence

Ce projet est sous licence MIT. Voir le fichier LICENSE pour plus de détails.
👤 Auteur

Dreipfelt

    GitHub: @Dreipfelt

🙏 Remerciements

    Projet OASIS et ses contributeurs
    data.gouv.fr pour les données ouvertes
    La communauté Streamlit

<p align="center"> Développé avec ❤️ dans le cadre d'une formation Data Science </p> ```