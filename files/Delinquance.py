import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import json
import requests

st.set_page_config(
    page_title="Oasis - Crime & Delinquency",
    page_icon=":rotating_light:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTES
# ══════════════════════════════════════════════════════════════════════════════
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

REG_NAMES = {
    'R11': 'Île-de-France',      'R24': 'Centre-Val de Loire',
    'R27': 'Bourgogne-FC',       'R28': 'Normandie',
    'R32': 'Hauts-de-France',    'R44': 'Grand Est',
    'R52': 'Pays de la Loire',   'R53': 'Bretagne',
    'R75': 'Nouvelle-Aquitaine', 'R76': 'Occitanie',
    'R84': 'Auvergne-RA',        'R93': 'PACA',
    'R94': 'Corse',
    'R01': 'Guadeloupe', 'R02': 'Martinique', 'R03': 'Guyane',
    'R04': 'La Réunion',  'R06': 'Mayotte',
}
METRO_REGIONS = [
    'R11','R24','R27','R28','R32','R44',
    'R52','R53','R75','R76','R84','R93','R94',
]
MODELE_COLORS = {
    'Prophet': '#EF553B', 'XGBoost': '#636EFA', 'LightGBM': '#00CC96',
}

# ══════════════════════════════════════════════════════════════════════════════
# CHARGEMENT
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data
def load_data():
    hist  = pd.read_csv(os.path.join(DATA_DIR, 'delinquance_region.csv'))
    prev  = pd.read_csv(os.path.join(DATA_DIR, 'delinquance_previsions_2030.csv'))
    bench = pd.read_csv(os.path.join(DATA_DIR, 'delinquance_benchmark.csv'))
    return hist, prev, bench

@st.cache_data
def load_geojson():
    local = os.path.join(DATA_DIR, 'regions.geojson')
    if os.path.exists(local):
        with open(local) as f:
            return json.load(f)
    url = (
        "https://raw.githubusercontent.com/gregoiredavid/"
        "france-geojson/master/regions-version-simplifiee.geojson"
    )
    return requests.get(url, timeout=10).json()

with st.spinner("Loading data and preparing maps..."):
    df_hist, df_prev, df_bench = load_data()
    geojson_regions = load_geojson()

ALL_CRIMES  = sorted(df_hist['indicateur'].dropna().unique())
ALL_REGIONS = sorted(
    [r for r in df_hist['region'].dropna().unique() if r in METRO_REGIONS],
    key=lambda r: REG_NAMES.get(r, r)
)
YEARS_HIST = sorted(df_hist['annee'].dropna().unique().astype(int))


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def make_choropleth(df_map, color_col, title, colorscale='Reds',
                    label='Nb of facts', opacity=0.7, range_color=None):
    """Carte choroplèthe statique (une seule année)."""
    df_plot = df_map.copy()
    df_plot['code_insee'] = df_plot['region'].str.replace('R', '', regex=False)
    kwargs = dict(
        geojson=geojson_regions,
        locations='code_insee',
        featureidkey='properties.code',
        color=color_col,
        color_continuous_scale=colorscale,
        hover_name='region_label',
        hover_data={color_col: ':,.1f', 'code_insee': False},
        mapbox_style='carto-positron',
        zoom=4.5,
        center={'lat': 46.5, 'lon': 2.5},
        opacity=opacity,
        title=title,
        labels={color_col: label},
    )
    if range_color:
        kwargs['range_color'] = range_color
    fig = px.choropleth_mapbox(df_plot, **kwargs)
    fig.update_layout(
        height=520,
        margin=dict(l=0, r=0, t=40, b=0),
        coloraxis_colorbar=dict(title=label),
    )
    return fig


def make_animated_choropleth(df_all_years, color_col, title,
                              colorscale='Reds', label='Nb of facts',
                              opacity=0.7, range_color=None):
    """
    Carte choroplèthe ANIMÉE.

    Clé du fonctionnement : chaque région doit être présente dans CHAQUE
    frame. On force cela avec un reindex complet (produit cartésien
    années × régions) avant de passer à Plotly.
    """
    df = df_all_years.copy()
    df['code_insee']   = df['region'].str.replace('R', '', regex=False)
    df['region_label'] = df['region'].map(REG_NAMES)

    # ── Garantir que toutes les régions sont présentes dans chaque frame ──
    all_years   = sorted(df['annee'].unique())
    all_regions = df['region'].unique()
    index_full  = pd.MultiIndex.from_product(
        [all_years, all_regions], names=['annee', 'region']
    )
    df_full = (
        df.set_index(['annee', 'region'])
          .reindex(index_full, fill_value=0)
          .reset_index()
    )
    # Re-remplir les colonnes dérivées après reindex
    df_full['code_insee']   = df_full['region'].str.replace('R', '', regex=False)
    df_full['region_label'] = df_full['region'].map(REG_NAMES)
    df_full['annee_str']    = df_full['annee'].astype(str)

    if range_color is None:
        range_color = [0, df_full[color_col].max()]

    fig = px.choropleth_mapbox(
        df_full.sort_values('annee'),
        geojson=geojson_regions,
        locations='code_insee',
        featureidkey='properties.code',
        color=color_col,
        color_continuous_scale=colorscale,
        range_color=range_color,
        hover_name='region_label',
        hover_data={color_col: ':,.0f', 'code_insee': False, 'annee_str': False},
        animation_frame='annee_str',
        animation_group='code_insee',
        category_orders={'annee_str': [str(y) for y in all_years]},
        mapbox_style='carto-positron',
        zoom=4.5,
        center={'lat': 46.5, 'lon': 2.5},
        opacity=opacity,
        title=title,
        labels={color_col: label},
    )
    fig.update_layout(
        height=580,
        margin=dict(l=0, r=0, t=40, b=0),
        coloraxis_colorbar=dict(title=label),
    )
    # Vitesse de l'animation : 800 ms par frame
    fig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 800
    fig.layout.updatemenus[0].buttons[0].args[1]['transition']['duration'] = 300
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.image(
        "https://raw.githubusercontent.com/nclsprsnw/oasis/refs/heads/main/docs/images/oasis_logo.png",
        width=120,
    )
    st.markdown("## 🚨 Crime & Delinquency")
    st.markdown("---")
    section = st.radio(
        "Navigate to",
        options=[
            "🗺️ An overview of crime in France",
            "📈 Top 7 Regions with Highest Variation",
            "🏆 Top & Bottom 5 Regions",
            "📊 Historical by Region",
            "⚠️ Crime Risk Scores by Region",
            "🔮 Predictions 2022-2030",
            "📋 Model Performance",
        ],
        label_visibility="collapsed",
        key="nav_section",
    )
    st.markdown("---")
    st.caption("Data: Police Nationale + Gendarmerie Nationale · 2012-2021")

st.header("Crime & Delinquency in France (2012-2030)")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — AN OVERVIEW (animée)
# ══════════════════════════════════════════════════════════════════════════════
if section == "🗺️ An overview of crime in France":
    st.subheader("An overview of crime in France", divider=True)
    st.write(
        "This map shows the total number of recorded crimes per region. "
        "Press **Play** or drag the slider below the map to animate 2012–2021."
    )

    crime_mode = st.radio(
        "Crime selection",
        ["All crimes", "Select specific crimes"],
        horizontal=True,
        key="overview_mode",
    )
    if crime_mode == "Select specific crimes":
        crimes_sel = st.multiselect(
            "Select crimes / offences", ALL_CRIMES,
            default=ALL_CRIMES[:3], key="crimes_overview",
        )
    else:
        crimes_sel = ALL_CRIMES

    # Agréger toutes les années
    df_anim = (
        df_hist[
            (df_hist['indicateur'].isin(crimes_sel)) &
            (df_hist['region'].isin(METRO_REGIONS))
        ]
        .groupby(['annee', 'region'])['nb_faits_A'].sum()
        .reset_index()
    )

    fig_overview = make_animated_choropleth(
        df_anim, 'nb_faits_A',
        title="Total recorded crimes by region",
        colorscale='Reds',
        label='Nb of facts',
    )
    st.plotly_chart(fig_overview, use_container_width=True)
    st.write(
        "Colour intensity reflects the total number of recorded facts. "
        "Regions with no data appear in light grey."
    )

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — TOP 7 VARIATION
# ══════════════════════════════════════════════════════════════════════════════
elif section == "📈 Top 7 Regions with Highest Variation":
    st.subheader("Top 7 Regions with Highest Variation", divider=True)
    st.write("Select the period and the crimes to view the top 7 regions with the highest variation.")

    left_co, right_co = st.columns(2)
    with left_co:
        selected_year_1 = st.selectbox(
            "Select Year 1", options=YEARS_HIST, index=0, key="year_select_1"
        )
    with right_co:
        selected_year_2 = st.selectbox(
            "Select Year 2", options=YEARS_HIST,
            index=len(YEARS_HIST)-1, key="year_select_2"
        )

    crime_mode_var = st.radio(
        "Crime selection", ["All crimes", "Select specific crimes"],
        horizontal=True, key="var_mode",
    )
    if crime_mode_var == "Select specific crimes":
        crimes_var = st.multiselect(
            "Select crimes / offences", ALL_CRIMES,
            default=ALL_CRIMES[:3], key="crimes_var",
        )
    else:
        crimes_var = ALL_CRIMES

    def get_region_total(year, crimes):
        return (
            df_hist[
                (df_hist['annee'] == year) &
                (df_hist['indicateur'].isin(crimes)) &
                (df_hist['region'].isin(METRO_REGIONS))
            ]
            .groupby('region')['nb_faits_A'].sum()
            .reset_index()
            .rename(columns={'nb_faits_A': f'total_{year}'})
        )

    df_y1 = get_region_total(selected_year_1, crimes_var)
    df_y2 = get_region_total(selected_year_2, crimes_var)
    variation_data = pd.merge(df_y1, df_y2, on='region', how='outer').fillna(0)
    variation_data['difference'] = (
        variation_data[f'total_{selected_year_2}'] - variation_data[f'total_{selected_year_1}']
    )
    variation_data['variation_%'] = np.where(
        variation_data[f'total_{selected_year_1}'] > 0,
        variation_data['difference'] / variation_data[f'total_{selected_year_1}'] * 100,
        0,
    )
    variation_data['region_label'] = variation_data['region'].map(REG_NAMES)
    top7 = variation_data.sort_values('difference', ascending=False).head(7)

    fig_var = px.bar(
        top7.sort_values('variation_%', ascending=True),
        x='region_label', y='variation_%',
        title=f"Top 7 Regions by Crime Variation ({selected_year_1} to {selected_year_2})",
        labels={'region_label': 'Region', 'variation_%': 'Variation (%)'},
        color='variation_%',
        color_continuous_scale=px.colors.sequential.Reds,
    )
    st.plotly_chart(fig_var, use_container_width=True)

    display_top7 = top7[[
        'region_label', 'region',
        f'total_{selected_year_1}', f'total_{selected_year_2}',
        'difference', 'variation_%',
    ]].copy()
    display_top7.columns = [
        'Region', 'Code',
        f'Total {selected_year_1}', f'Total {selected_year_2}',
        'Abs. Change', 'Change (%)',
    ]
    display_top7['Change (%)'] = display_top7['Change (%)'].map('{:,.2f}%'.format)
    st.dataframe(display_top7, hide_index=True, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — TOP & BOTTOM 5
# ══════════════════════════════════════════════════════════════════════════════
elif section == "🏆 Top & Bottom 5 Regions":
    st.subheader("Top & Bottom 5 Regions", divider=True)
    st.write("Select a year and the crimes to compare the most and least affected regions.")

    col1, col2 = st.columns(2)
    with col1:
        year_tb = st.selectbox(
            "Select Year", options=YEARS_HIST,
            index=len(YEARS_HIST)-1, key="year_topbottom"
        )
    with col2:
        crime_mode_tb = st.radio(
            "Crime selection", ["All crimes", "Select specific crimes"],
            horizontal=True, key="tb_mode",
        )

    if crime_mode_tb == "Select specific crimes":
        crimes_tb = st.multiselect(
            "Select crimes / offences", ALL_CRIMES,
            default=ALL_CRIMES[:3], key="crimes_tb",
        )
    else:
        crimes_tb = ALL_CRIMES

    df_tb = (
        df_hist[
            (df_hist['annee'] == year_tb) &
            (df_hist['indicateur'].isin(crimes_tb)) &
            (df_hist['region'].isin(METRO_REGIONS))
        ]
        .groupby('region')['nb_faits_A'].sum()
        .reset_index()
        .sort_values('nb_faits_A', ascending=False)
    )
    df_tb['region_label'] = df_tb['region'].map(REG_NAMES)

    combined = pd.concat([
        df_tb.head(5).assign(category='Top 5'),
        df_tb.tail(5).assign(category='Bottom 5'),
    ])

    fig_tb = px.bar(
        combined.sort_values('nb_faits_A', ascending=True),
        x='nb_faits_A', y='region_label',
        color='category', orientation='h',
        title=f"Top & Bottom 5 Regions — {year_tb}",
        labels={'nb_faits_A': 'Nb of facts', 'region_label': 'Region', 'category': ''},
        color_discrete_map={'Top 5': '#EF553B', 'Bottom 5': '#00CC96'},
        text='nb_faits_A',
    )
    fig_tb.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
    fig_tb.update_layout(height=400, plot_bgcolor='#fafafa')
    st.plotly_chart(fig_tb, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — HISTORICAL BY REGION
# ══════════════════════════════════════════════════════════════════════════════
elif section == "📊 Historical by Region":
    st.subheader("Select Region(s) to View Historical Crimes", divider=True)
    st.write(
        "Select one or more regions and the crimes to explore the historical "
        "evolution from 2012 to 2021."
    )

    col_r, col_c = st.columns(2)
    with col_r:
        regions_sel = st.multiselect(
            "Select Region(s)", ALL_REGIONS,
            default=ALL_REGIONS[:2],
            format_func=lambda r: REG_NAMES.get(r, r),
            key="regions_hist",
        )
    with col_c:
        crime_mode_hist = st.radio(
            "Crime selection", ["All crimes", "Select specific crimes"],
            horizontal=True, key="hist_mode",
        )

    if crime_mode_hist == "Select specific crimes":
        crimes_hist = st.multiselect(
            "Select crimes / offences", ALL_CRIMES,
            default=ALL_CRIMES[:3], key="crimes_hist",
        )
    else:
        crimes_hist = ALL_CRIMES

    if regions_sel:
        df_hist_sel = (
            df_hist[
                (df_hist['region'].isin(regions_sel)) &
                (df_hist['indicateur'].isin(crimes_hist))
            ]
            .groupby(['annee', 'region'])['nb_faits_A'].sum()
            .reset_index()
        )
        df_hist_sel['region_label'] = df_hist_sel['region'].map(REG_NAMES)

        fig_line = px.line(
            df_hist_sel.sort_values('annee'),
            x='annee', y='nb_faits_A', color='region_label',
            markers=True,
            title="Historical evolution of recorded crimes by region",
            labels={
                'annee': 'Year', 'nb_faits_A': 'Nb of facts',
                'region_label': 'Region'
            },
        )
        fig_line.update_layout(height=420, hovermode='x unified')
        st.plotly_chart(fig_line, use_container_width=True)
    else:
        st.info("Select at least one region to display the chart.")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — CRIME RISK SCORES (animée)
# ══════════════════════════════════════════════════════════════════════════════
elif section == "⚠️ Crime Risk Scores by Region":
    st.subheader("Crime Risk Scores by Region", divider=True)
    st.write(
        "Crime intensity score normalised 0–100 per year. "
        "Press **Play** or drag the slider to animate 2012–2021."
    )

    crime_mode_risk = st.radio(
        "Crime selection", ["All crimes", "Select specific crimes"],
        horizontal=True, key="risk_mode",
    )
    if crime_mode_risk == "Select specific crimes":
        crimes_risk = st.multiselect(
            "Select crimes / offences", ALL_CRIMES,
            default=ALL_CRIMES[:5], key="crimes_risk",
        )
    else:
        crimes_risk = ALL_CRIMES

    # Agréger toutes les années
    df_risk_all = (
        df_hist[
            (df_hist['indicateur'].isin(crimes_risk)) &
            (df_hist['region'].isin(METRO_REGIONS))
        ]
        .groupby(['annee', 'region'])['nb_faits_A'].sum()
        .reset_index()
    )

    # Score 0-100 normalisé ANNÉE PAR ANNÉE
    scores = []
    for yr in sorted(df_risk_all['annee'].unique()):
        sub = df_risk_all[df_risk_all['annee'] == yr].copy()
        vmin = sub['nb_faits_A'].min()
        vmax = sub['nb_faits_A'].max()
        sub['score'] = np.where(
            vmax > vmin,
            (sub['nb_faits_A'] - vmin) / (vmax - vmin) * 100,
            50,
        ).round(1)
        scores.append(sub)
    df_risk_scored = pd.concat(scores, ignore_index=True)

    fig_risk_anim = make_animated_choropleth(
        df_risk_scored, 'score',
        title="Crime Risk Score by Region (0 = lowest, 100 = highest)",
        colorscale='RdYlGn_r',
        label='Risk Score (0-100)',
        range_color=[0, 100],
    )
    st.plotly_chart(fig_risk_anim, use_container_width=True)

    # Barres + heatmap sur l'année sélectionnée
    year_risk = st.select_slider(
        "Select a year for detailed view",
        options=YEARS_HIST, value=YEARS_HIST[-1], key="year_risk"
    )
    df_risk_yr = df_risk_scored[df_risk_scored['annee'] == year_risk].copy()
    df_risk_yr['region_label'] = df_risk_yr['region'].map(REG_NAMES)

    fig_risk_bar = px.bar(
        df_risk_yr.sort_values('score', ascending=True),
        x='score', y='region_label', orientation='h',
        color='score', color_continuous_scale='RdYlGn_r',
        title=f"Crime Risk Score ranking — {year_risk}",
        labels={'score': 'Risk Score (0-100)', 'region_label': 'Region'},
        text='score',
    )
    fig_risk_bar.update_traces(texttemplate='%{text:.1f}', textposition='outside')
    fig_risk_bar.update_layout(
        height=420, coloraxis_showscale=False, plot_bgcolor='#fafafa'
    )
    st.plotly_chart(fig_risk_bar, use_container_width=True)

    if crime_mode_risk == "Select specific crimes" and len(crimes_risk) > 1:
        st.write("**Crime breakdown by region**")
        pivot_risk = (
            df_hist[
                (df_hist['annee'] == year_risk) &
                (df_hist['indicateur'].isin(crimes_risk)) &
                (df_hist['region'].isin(METRO_REGIONS))
            ]
            .groupby(['region', 'indicateur'])['nb_faits_A'].sum()
            .unstack('indicateur').fillna(0)
        )
        pivot_risk.index = pivot_risk.index.map(REG_NAMES)
        pivot_risk.columns = [c[:35] for c in pivot_risk.columns]
        fig_heat = px.imshow(
            pivot_risk, color_continuous_scale='Reds',
            labels=dict(color='Nb of facts'),
            title=f"Crime profile by region — {year_risk}",
            aspect='auto', text_auto='.0f',
        )
        fig_heat.update_layout(height=420)
        st.plotly_chart(fig_heat, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — PREDICTIONS 2022-2030
# ══════════════════════════════════════════════════════════════════════════════
elif section == "🔮 Predictions 2022-2030":
    st.subheader("Predictions 2022-2030", divider=True)
    st.write(
        "Forecasts using Prophet, XGBoost, and LightGBM — "
        "trained on 2012-2019, validated on 2020-2021."
    )

    col_p1, col_p2 = st.columns(2)
    with col_p1:
        region_pred = st.selectbox(
            "Select Region", ALL_REGIONS,
            format_func=lambda r: REG_NAMES.get(r, r), key="region_pred"
        )
    with col_p2:
        crime_pred = st.selectbox(
            "Select Crime / Offence", ALL_CRIMES, key="crime_pred"
        )

    modeles_pred = st.multiselect(
        "Models to display", ['Prophet', 'XGBoost', 'LightGBM'],
        default=['Prophet', 'XGBoost', 'LightGBM'], key="modeles_pred"
    )

    hist_pred = (
        df_hist[
            (df_hist['indicateur'] == crime_pred) &
            (df_hist['region']     == region_pred)
        ]
        .groupby('annee')['nb_faits_A'].sum()
        .reset_index()
    )

    fig_pred = go.Figure()
    fig_pred.add_trace(go.Scatter(
        x=hist_pred['annee'], y=hist_pred['nb_faits_A'],
        mode='lines+markers', name='Historical (2012-2021)',
        line=dict(color='#2c3e50', width=3), marker=dict(size=8),
    ))

    for modele in modeles_pred:
        prev_serie = df_prev[
            (df_prev['indicateur']    == crime_pred) &
            (df_prev['region']        == region_pred) &
            (df_prev['source_modele'] == modele) &
            (df_prev['annee']         >= 2022)
        ].sort_values('annee')

        if len(prev_serie) == 0:
            continue

        color = MODELE_COLORS.get(modele, '#999')

        if modele == 'Prophet' and prev_serie['yhat_lower'].notna().any():
            fig_pred.add_trace(go.Scatter(
                x=pd.concat([prev_serie['annee'], prev_serie['annee'][::-1]]),
                y=pd.concat([prev_serie['yhat_upper'], prev_serie['yhat_lower'][::-1]]),
                fill='toself',
                fillcolor=f'rgba{tuple(list(px.colors.hex_to_rgb(color)) + [0.12])}',
                line=dict(color='rgba(0,0,0,0)'),
                showlegend=False, name=f'{modele} CI',
            ))

        fig_pred.add_trace(go.Scatter(
            x=prev_serie['annee'], y=prev_serie['yhat'],
            mode='lines+markers', name=f'{modele} forecast',
            line=dict(color=color, width=2.5, dash='dash'),
            marker=dict(size=7),
        ))

    fig_pred.add_vline(x=2021.5, line_dash='dot', line_color='gray', opacity=0.5)
    fig_pred.add_annotation(
        x=2022, y=1, yref='paper',
        text='← Historical | Forecast →',
        showarrow=False, font=dict(color='gray', size=11),
    )
    fig_pred.update_layout(
        height=480,
        title=f"{crime_pred[:60]} — {REG_NAMES.get(region_pred, region_pred)}",
        xaxis_title='Year', yaxis_title='Nb of facts',
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        plot_bgcolor='#fafafa',
    )
    st.plotly_chart(fig_pred, use_container_width=True)

    prev_table = df_prev[
        (df_prev['indicateur']    == crime_pred) &
        (df_prev['region']        == region_pred) &
        (df_prev['source_modele'].isin(modeles_pred)) &
        (df_prev['annee']         >= 2022)
    ].pivot_table(
        index='annee', columns='source_modele', values='yhat', aggfunc='first'
    ).round(0).reset_index()
    prev_table.columns.name = None
    st.dataframe(
        prev_table, hide_index=True, use_container_width=True,
        column_config={'annee': 'Year'}
    )

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
elif section == "📋 Model Performance":
    st.subheader("Model Performance (test on 2020-2021 actuals)", divider=True)
    st.write(
        "Models trained on 2012-2019, tested on 2020-2021. "
        "MAPE = mean absolute percentage error (lower is better)."
    )

    if len(df_bench) > 0:
        summary = (
            df_bench.groupby('modele')[['RMSE', 'MAE', 'MAPE']]
            .mean().round(2).sort_values('MAPE').reset_index()
        )
        summary.columns = ['Model', 'RMSE', 'MAE', 'MAPE (%)']

        col_b1, col_b2 = st.columns([1, 2])
        with col_b1:
            st.dataframe(summary, hide_index=True, use_container_width=True)
        with col_b2:
            fig_bench = px.bar(
                summary.sort_values('MAPE (%)', ascending=True),
                x='Model', y='MAPE (%)',
                color='Model',
                color_discrete_map={
                    'Prophet': '#EF553B', 'XGBoost': '#636EFA', 'LightGBM': '#00CC96'
                },
                title='MAPE by model (lower = better)',
                text='MAPE (%)',
            )
            fig_bench.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig_bench.update_layout(showlegend=False, height=320)
            st.plotly_chart(fig_bench, use_container_width=True)

        if 'indicateur' in df_bench.columns and 'modele' in df_bench.columns:
            st.subheader("MAPE by crime type and model", divider=True)
            pivot_mape = (
                df_bench.groupby(['indicateur', 'modele'])['MAPE']
                .mean().round(1).unstack('modele').fillna(0)
            )
            pivot_mape.index = [x[:40] for x in pivot_mape.index]
            fig_heat_bench = px.imshow(
                pivot_mape, color_continuous_scale='RdYlGn_r',
                labels=dict(color='MAPE (%)'),
                title="MAPE (%) — greener = more accurate",
                aspect='auto', text_auto='.1f',
            )
            fig_heat_bench.update_layout(height=420)
            st.plotly_chart(fig_heat_bench, use_container_width=True)
    else:
        st.info("No benchmark data available.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Sources: French Ministry of Interior — "
    "Police Nationale (PN) + Gendarmerie Nationale (GN) | "
    "Historical data: 2012-2021 | Forecasts: Prophet, XGBoost, LightGBM | "
    "Aggregation: department → region (INSEE 2016 boundaries)"
)
