# CrimesEtDelits.py
import streamlit as st
import pandas as pd
import plotly.express as px
import os

class CrimeMapper:
    def __init__(self):
        self.df = None
        self.df_geo = None
        self.crime_columns = []
        self.dept_col = None  # Colonne identifiant département

    def load_data(self, file_path):
        """Charge les données Excel avec sélection de feuille"""
        try:
            if not os.path.exists(file_path):
                st.error(f"Fichier '{file_path}' non trouvé.")
                return False

            xls = pd.ExcelFile(file_path)
            st.write("Feuilles disponibles :", xls.sheet_names)

            # Choix de la feuille
            sheet_name = st.selectbox("Sélectionner la feuille à charger", xls.sheet_names)
            self.df = pd.read_excel(file_path, sheet_name=sheet_name)
            st.success(f"Feuille '{sheet_name}' chargée : {self.df.shape[0]} lignes, {self.df.shape[1]} colonnes")

            # Afficher les colonnes disponibles
            st.write("Colonnes disponibles :", list(self.df.columns))

            # Détection automatique de la colonne département
            candidates = ['Libellé index', 'Libelle index', 'Libellé crime et délit', 'Libelle crime et delit']
            found = None
            for c in candidates:
                if c in self.df.columns:
                    found = c
                    break

            if not found:
                st.error("❌ Aucune colonne département trouvée parmi : " + ", ".join(candidates))
                return False

            self.dept_col = found
            st.info(f"Colonne département utilisée : '{self.dept_col}'")

            self.prepare_data()
            return True

        except Exception as e:
            st.error(f"Erreur lors du chargement : {e}")
            return False

    def prepare_data(self):
        """Prépare les colonnes de crimes et les données géographiques"""
        # Colonnes de crimes : toutes sauf 'Année' et colonne département
        self.crime_columns = [col for col in self.df.columns if col not in ['Année', self.dept_col]]

        # Préparation géographique
        self.df_geo = self.prepare_geographic_data()

    def prepare_geographic_data(self):
        """Ajoute lat/lon simulées pour les départements français"""
        # Coordonnées simplifiées pour l'exemple
        dept_coords = {
            1: {'lat': 46.2044, 'lon': 5.2255, 'name': 'Ain'},
            2: {'lat': 49.5700, 'lon': 3.6200, 'name': 'Aisne'},
            75: {'lat': 48.8566, 'lon': 2.3522, 'name': 'Paris'},
            # Compléter avec tous les départements si nécessaire
        }

        df_geo = self.df.copy()

        def safe_int(x):
            try:
                return int(x)
            except:
                return x

        df_geo['dept_code'] = df_geo[self.dept_col].apply(safe_int)
        df_geo['lat'] = df_geo['dept_code'].map(lambda x: dept_coords.get(x, {}).get('lat', 46.8))
        df_geo['lon'] = df_geo['dept_code'].map(lambda x: dept_coords.get(x, {}).get('lon', 2.3))
        df_geo['dept_name'] = df_geo['dept_code'].map(lambda x: dept_coords.get(x, {}).get('name', f'Région {x}'))

        return df_geo

    def create_map(self, year, crime_type):
        df_filtered = self.df_geo[self.df_geo['Année'] == year].copy()
        df_filtered['crime_count'] = pd.to_numeric(df_filtered.get(crime_type, 0), errors='coerce').fillna(0)

        fig = px.scatter_mapbox(
            df_filtered,
            lat='lat',
            lon='lon',
            size='crime_count',
            color='crime_count',
            hover_name='dept_name',
            hover_data={'Année': True, 'crime_count': ':.0f', 'lat': False, 'lon': False},
            color_continuous_scale='Reds',
            size_max=50,
            zoom=5,
            center={'lat': 46.8, 'lon': 2.3},
            mapbox_style='open-street-map',
            title=f'Répartition de {crime_type} en {year}'
        )
        return fig

    def create_time_series(self, crime_type):
        df_yearly = self.df.groupby('Année')[crime_type].sum().reset_index()
        df_yearly[crime_type] = pd.to_numeric(df_yearly[crime_type], errors='coerce').fillna(0)

        fig = px.line(
            df_yearly,
            x='Année',
            y=crime_type,
            markers=True,
            title=f'Évolution de {crime_type} (2012-2021)'
        )
        return fig

# --- STREAMLIT APP ---
st.title("📊 Visualisation des Crimes en France (2012-2021)")

mapper = CrimeMapper()

# Chemin relatif depuis le dossier racine du projet
file_path = st.text_input("Chemin vers le fichier Excel", "data/CrimesEtDelits_from2012to2021.xlsx")

if mapper.load_data(file_path):
    year = st.slider(
        "Sélectionner l'année",
        int(mapper.df['Année'].min()),
        int(mapper.df['Année'].max()),
        int(mapper.df['Année'].max())
    )
    crime_type = st.selectbox("Sélectionner le type de crime", mapper.crime_columns)

    st.subheader("Carte interactive")
    st.plotly_chart(mapper.create_map(year, crime_type), use_container_width=True)

    st.subheader("Série temporelle")
    st.plotly_chart(mapper.create_time_series(crime_type), use_container_width=True)
