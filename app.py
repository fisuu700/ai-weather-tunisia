"""
app.py
======
Plateforme d'intelligence météorologique (dashboard Streamlit).

Fonctionnalités :
  - Localisation : ville prédéfinie, CARTE INTERACTIVE (clic Folium) ou
    coordonnées GPS manuelles ;
  - Unités : métrique (°C, km/h) ou impérial (°F, mph) ;
  - Prévisions MULTI-JOURS (1, 3, 5 ou 7 jours) de la température max/min,
    du risque de pluie, du vent, de l'humidité et de la pression ;
  - Intervalle de confiance (IC 90 %) autour de la température maximale ;
  - Vue HEURE PAR HEURE de la journée à venir + conditions actuelles
    temps réel (Open-Meteo) ;
  - Performances des modèles, importances des variables (tmax ET pluie) ;
  - Explicabilité dynamique SHAP (XAI) pour un jour donné ;
  - Climatologie : normales mensuelles / saisonnières et anomalies annuelles.

Lancement :
    streamlit run app.py
"""

from __future__ import annotations

import os
import sys
from datetime import date, timedelta

import folium
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_folium import st_folium

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src import climatology                    # noqa: E402
from src import explain as explain_module      # noqa: E402
from src import predict as predict_module      # noqa: E402
from src import tounsi_ai                       # noqa: E402
from src import train as train_module          # noqa: E402
from src import units as U                     # noqa: E402
from src import weather_api                     # noqa: E402
from src.data_loader import VILLES, obtenir_info_geographique, REGIONS_TUNISIE  # noqa: E402

# --------------------------------------------------------------------------
# Configuration de la page
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Plateforme Météo & IA Tunisie — 24 Gouvernorats",
    page_icon="🌤️",
    layout="wide",
)

# --------------------------------------------------------------------------
# Design System & UI Styling (CSS Moderne)
# --------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}

/* Hero Banner */
.weather-hero {
    background: linear-gradient(135deg, rgba(30, 58, 138, 0.16) 0%, rgba(14, 165, 233, 0.12) 50%, rgba(16, 185, 129, 0.08) 100%);
    border: 1px solid rgba(56, 189, 248, 0.28);
    border-radius: 20px;
    padding: 24px 28px;
    margin-bottom: 22px;
    backdrop-filter: blur(12px);
    box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.12);
}

.weather-hero-title {
    font-size: 2.1rem;
    font-weight: 800;
    margin-bottom: 6px;
    background: linear-gradient(135deg, #0284c7 0%, #2563eb 50%, #7c3aed 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    display: flex;
    align-items: center;
    gap: 12px;
}

.weather-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 14px;
    border-radius: 9999px;
    font-size: 0.83rem;
    font-weight: 600;
    margin-right: 8px;
    margin-top: 8px;
}

.badge-active {
    background: rgba(16, 185, 129, 0.15);
    color: #10b981;
    border: 1px solid rgba(16, 185, 129, 0.35);
}

.badge-info {
    background: rgba(14, 165, 233, 0.15);
    color: #0284c7;
    border: 1px solid rgba(14, 165, 233, 0.35);
}

.badge-warning {
    background: rgba(245, 158, 11, 0.15);
    color: #f59e0b;
    border: 1px solid rgba(245, 158, 11, 0.35);
}

/* Carte Météo Temps Réel (Live Hero Card) */
.live-hero-card {
    background: linear-gradient(135deg, rgba(14, 165, 233, 0.12) 0%, rgba(30, 58, 138, 0.16) 50%, rgba(99, 102, 241, 0.10) 100%);
    border: 1px solid rgba(56, 189, 248, 0.32);
    border-radius: 20px;
    padding: 22px 26px;
    margin-bottom: 22px;
    backdrop-filter: blur(14px);
    box-shadow: 0 10px 28px -6px rgba(0, 0, 0, 0.12);
}

.live-hero-header {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 12px;
    margin-bottom: 16px;
}

.live-hero-body {
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 16px;
}

.live-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(16, 185, 129, 0.15);
    border: 1px solid rgba(16, 185, 129, 0.35);
    color: #10b981;
    border-radius: 9999px;
    padding: 3px 12px;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    margin-bottom: 6px;
    display: inline-flex;
}

.live-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #10b981;
    display: inline-block;
    animation: pulse-dot 1.8s ease-in-out infinite;
}

@keyframes pulse-dot {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.5; transform: scale(0.75); }
}

.live-temp-display {
    font-size: 3.5rem;
    font-weight: 800;
    line-height: 1;
    background: linear-gradient(180deg, #ffffff 0%, #cbd5e1 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -1px;
    margin-bottom: 4px;
}

.live-hero-stats {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    align-items: center;
}

.live-stat-chip {
    background: rgba(255, 255, 255, 0.06);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 10px;
    padding: 6px 14px;
    font-size: 0.88rem;
    display: inline-flex;
    align-items: center;
    gap: 6px;
    transition: all 0.2s ease;
}

.live-stat-chip:hover {
    background: rgba(56, 189, 248, 0.10);
    border-color: rgba(56, 189, 248, 0.35);
}

/* Cartes Métriques (Streamlit metrics) */
div[data-testid="stMetric"] {
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 16px;
    padding: 18px 22px;
    box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.05);
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
}

div[data-testid="stMetric"]:hover {
    transform: translateY(-3px);
    border-color: rgba(56, 189, 248, 0.5);
    box-shadow: 0 12px 28px -5px rgba(56, 189, 248, 0.18);
}

/* Multi-day forecast cards */
.forecast-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
    gap: 12px;
    margin: 14px 0 22px 0;
}
.forecast-card {
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.09);
    border-radius: 16px;
    padding: 14px 10px;
    text-align: center;
    backdrop-filter: blur(10px);
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
}
.forecast-card:hover {
    transform: translateY(-4px);
    border-color: rgba(56, 189, 248, 0.5);
    background: rgba(56, 189, 248, 0.08);
    box-shadow: 0 10px 24px -6px rgba(0, 0, 0, 0.2);
}
.forecast-card.card-today {
    border-color: rgba(14, 165, 233, 0.6);
    background: linear-gradient(180deg, rgba(14, 165, 233, 0.14) 0%, rgba(30, 58, 138, 0.12) 100%);
    box-shadow: 0 6px 20px -4px rgba(14, 165, 233, 0.2);
}
.forecast-date {
    font-size: 0.82rem;
    font-weight: 700;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.forecast-icon {
    font-size: 2.1rem;
    margin: 6px 0;
    line-height: 1;
}
.forecast-temps {
    display: flex;
    justify-content: center;
    align-items: baseline;
    gap: 6px;
    margin-bottom: 6px;
}
.forecast-tmax {
    font-size: 1.2rem;
    font-weight: 800;
    color: #ef4444;
}
.forecast-tmin {
    font-size: 0.92rem;
    font-weight: 600;
    color: #60a5fa;
}
.forecast-badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 9999px;
    font-size: 0.72rem;
    font-weight: 600;
}

/* Chat Welcome Card */
.chat-welcome-card {
    background: linear-gradient(135deg, rgba(225, 29, 72, 0.09) 0%, rgba(244, 63, 94, 0.05) 50%, rgba(251, 146, 60, 0.05) 100%);
    border: 1px solid rgba(244, 63, 94, 0.28);
    border-radius: 18px;
    padding: 22px 26px;
    margin-bottom: 22px;
    box-shadow: 0 8px 24px -6px rgba(225, 29, 72, 0.08);
}

.chat-cat-header {
    font-size: 0.82rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: #e11d48;
    margin-bottom: 8px;
}

/* Onglets stylisés */
.stTabs [data-baseweb="tab-list"] {
    gap: 10px;
    border-bottom: 2px solid rgba(255, 255, 255, 0.08);
    margin-bottom: 18px;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 12px 12px 0 0;
    padding: 12px 22px;
    font-weight: 600;
    font-size: 0.95rem;
    transition: all 0.2s ease;
}

.stTabs [aria-selected="true"] {
    background: rgba(56, 189, 248, 0.12) !important;
    border-bottom: 3px solid #0284c7 !important;
    color: #0284c7 !important;
}

/* Boutons stylisés */
div.stButton > button {
    border-radius: 12px;
    font-weight: 600;
    transition: all 0.2s ease;
}

div.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px -3px rgba(0, 0, 0, 0.15);
}
</style>
""", unsafe_allow_html=True)

MODELES_DIR = train_module.MODELES_DIR
DATA_DIR = train_module.DATA_DIR
CIBLES_REQUIS = train_module.CIBLES_REQUIS
CIBLES_REGRESSION = train_module.CIBLES_REGRESSION
LIBELLES_CIBLES = {
    "tmax": "🌡️ Température max",
    "tmin": "🌡️ Température min",
    "rhum": "💧 Humidité",
    "wspd": "💨 Vent",
    "pres": "🧭 Pression",
    "rain": "🌧️ Risque de pluie",
}
MOIS_ORDRE = [climatology.MOIS_FR[m] for m in range(1, 13)]
JOURS_FR = ["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"]
MOIS_FR_COURT = ["", "Jan", "Fév", "Mar", "Avr", "Mai", "Juin", "Juil", "Août", "Sep", "Oct", "Nov", "Déc"]


# --------------------------------------------------------------------------
# Fonctions utilitaires (cache + formatage + styling)
# --------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def _charger_climatologie(lat: float, lon: float, debut: str) -> pd.DataFrame:
    """Historique long (normales de saison), réutilisé entre les recharges."""
    return climatology.charger_historique(lat, lon, debut=debut)


@st.cache_data(ttl=3600, show_spinner=False)
def _features_historique(lat: float, lon: float, debut: str, fin: str) -> pd.DataFrame:
    """Historique enrichi des features (pour l'explication SHAP d'un jour passé)."""
    from src.data_loader import fetch_data
    from src.feature_engineering import construire_features

    return construire_features(fetch_data(lat, lon, debut, fin, use_cache=True))


def _texte(valeur, format_spec: str = "{:.0f}", suffixe: str = "") -> str:
    """Formate une valeur numérique, ou « n/c » si manquante."""
    if valeur is None or (isinstance(valeur, float) and np.isnan(valeur)):
        return "n/c"
    return format_spec.format(valeur) + suffixe


def _couleur_pluie(proba_pct: float) -> str:
    if proba_pct >= 50:
        return "#DC3912"
    if proba_pct >= 25:
        return "#FFA500"
    return "#1C9078"


def _icone_meteo(wmo_code: int | None = None, proba_pluie: float | None = None, tmax: float | None = None) -> tuple[str, str]:
    """Retourne (emoji, description) adaptée selon WMO code ou prévision ML."""
    if wmo_code is not None:
        table = {
            0: ("☀️", "Ciel dégagé"),
            1: ("🌤️", "Principalement dégagé"),
            2: ("⛅", "Partiellement nuageux"),
            3: ("☁️", "Couvert"),
            45: ("🌫️", "Brouillard"),
            48: ("🌫️", "Brouillard givrant"),
            51: ("🌦️", "Bruine légère"),
            53: ("🌦️", "Bruine modérée"),
            55: ("🌧️", "Bruine dense"),
            61: ("🌧️", "Pluie faible"),
            63: ("🌧️", "Pluie modérée"),
            65: ("🌧️", "Forte pluie"),
            71: ("🌨️", "Neige faible"),
            73: ("🌨️", "Neige modérée"),
            75: ("❄️", "Forte neige"),
            80: ("🌦️", "Averses faibles"),
            81: ("🌧️", "Averses modérées"),
            82: ("⛈️", "Averses violentes"),
            95: ("⛈️", "Orage"),
            96: ("⛈️", "Orage avec grêle"),
            99: ("⛈️", "Orage violent"),
        }
        if wmo_code in table:
            return table[wmo_code]

    if proba_pluie is not None:
        if proba_pluie >= 0.55:
            return ("🌧️", "Pluie probable")
        if proba_pluie >= 0.30:
            return ("🌦️", "Risque d'averses")
        if tmax is not None and tmax >= 34:
            return ("☀️", "Très chaud & Ensoleillé")
        if tmax is not None and tmax >= 24:
            return ("🌤️", "Beau temps ensoleillé")
        return ("⛅", "Variable / Éclaircies")

    return ("🌤️", "Temps variable")


def _badge_uv(uv: float | None) -> tuple[str, str, str]:
    """Retourne (libellé, couleur_texte, couleur_fond) selon l'indice UV."""
    if uv is None or (isinstance(uv, float) and np.isnan(uv)):
        return ("n/c", "#94a3b8", "rgba(148, 163, 184, 0.15)")
    if uv < 3:
        return ("Faible", "#10b981", "rgba(16, 185, 129, 0.18)")
    if uv < 6:
        return ("Modéré", "#f59e0b", "rgba(245, 158, 11, 0.18)")
    if uv < 8:
        return ("Élevé", "#f97316", "rgba(249, 115, 22, 0.18)")
    if uv < 11:
        return ("Très élevé", "#ef4444", "rgba(239, 68, 68, 0.18)")
    return ("Extrême", "#a855f7", "rgba(168, 85, 247, 0.22)")


def _ressenti_temperature(t: float | None, wspd: float | None, rhum: float | None) -> float | None:
    """Calcule une estimation de la température ressentie (°C)."""
    if t is None:
        return None
    try:
        e = (rhum / 100.0) * 6.105 * np.exp((17.27 * t) / (237.7 + t)) if rhum is not None else 10.0
        v = (wspd / 3.6) if wspd is not None else 3.0
        at = t + 0.33 * e - 0.70 * v - 4.0
        return round(float(at), 1)
    except Exception:
        return round(float(t), 1)


def _styliser_figure(fig: go.Figure, hauteur: int = 360) -> go.Figure:
    """Harmonise le style d'un graphique Plotly avec le design system moderne."""
    fig.update_layout(
        height=hauteur,
        margin=dict(l=24, r=24, t=36, b=24),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Plus Jakarta Sans, sans-serif"),
        hoverlabel=dict(font_size=12, font_family="Plus Jakarta Sans"),
    )
    fig.update_xaxes(
        gridcolor="rgba(128, 128, 128, 0.12)",
        zerolinecolor="rgba(128, 128, 128, 0.15)",
    )
    fig.update_yaxes(
        gridcolor="rgba(128, 128, 128, 0.12)",
        zerolinecolor="rgba(128, 128, 128, 0.15)",
    )
    return fig


# --------------------------------------------------------------------------
# Barre latérale : paramètres
# --------------------------------------------------------------------------
st.sidebar.title("🌤️ Plateforme Météo")
st.sidebar.markdown("### 📍 Localisation")

mode_localisation = st.sidebar.radio(
    "Mode de sélection",
    ["Gouvernorat", "Carte interactive", "Coordonnées GPS"],
    index=0,
)

if mode_localisation == "Gouvernorat":
    region_filtre = st.sidebar.selectbox(
        "🗺️ Filtrer par région",
        ["Toutes les régions (24)"] + list(REGIONS_TUNISIE.keys()),
        index=0,
    )
    if region_filtre == "Toutes les régions (24)":
        liste_villes = list(VILLES.keys())
    else:
        liste_villes = [v for v in REGIONS_TUNISIE[region_filtre] if v in VILLES]

    idx_defaut = 0
    if "derniere_ville_selectionnee" in st.session_state and st.session_state["derniere_ville_selectionnee"] in liste_villes:
        idx_defaut = liste_villes.index(st.session_state["derniere_ville_selectionnee"])
    elif "Tunis" in liste_villes:
        idx_defaut = liste_villes.index("Tunis")
    elif "Sousse" in liste_villes:
        idx_defaut = liste_villes.index("Sousse")

    ville = st.sidebar.selectbox(
        f"Gouvernorat ({len(liste_villes)} disponible{'s' if len(liste_villes) > 1 else ''})",
        liste_villes,
        index=idx_defaut,
        format_func=lambda v: f"{v} ({obtenir_info_geographique(v)['region']}) {'✅' if predict_module.modeles_complets(v) else '⏳'}",
    )
    st.session_state["derniere_ville_selectionnee"] = ville
    lat, lon = VILLES[ville]
    info_geo = obtenir_info_geographique(ville)
elif mode_localisation == "Coordonnées GPS":
    st.sidebar.caption("Ex. Tunis : 36.81 / 10.18 — Sousse : 35.83 / 10.64 — Sfax : 34.74 / 10.76")
    lat = st.sidebar.number_input("Latitude", -90.0, 90.0, 36.8065, step=0.0001, format="%.4f")
    lon = st.sidebar.number_input("Longitude", -180.0, 180.0, 10.1815, step=0.0001, format="%.4f")
    ville = f"GPS_{lat:.2f}_{lon:.2f}"
    info_geo = obtenir_info_geographique(ville)
else:
    lat = st.session_state.get("carte_lat", VILLES["Tunis"][0])
    lon = st.session_state.get("carte_lon", VILLES["Tunis"][1])
    ville = f"GPS_{lat:.2f}_{lon:.2f}"
    info_geo = obtenir_info_geographique(ville)
    st.sidebar.caption("Cliquez un point sur la carte (zone principale) pour charger ses prévisions.")

st.sidebar.markdown("### 📏 Unités")
systeme = st.sidebar.selectbox(
    "Système d'unités", [U.METRIQUE, U.IMPERIAL], format_func=lambda k: U.LIBELLES[k]
)

st.sidebar.markdown("### 📅 Horizon de prévision")
horizon = st.sidebar.radio("Nombre de jours", [1, 3, 5, 7], index=1, horizontal=True)

st.sidebar.markdown("### ⚙️ Entraînement")
annees_test = st.sidebar.slider(
    "Années réservées au test", 1, 5, 2,
    help="Les dernières N années servent de jeu de test (split temporel).",
)
st.sidebar.caption("Entraîne 6 modèles : tmax, tmin, humidité, vent, pression + risque de pluie.")
bouton_train = st.sidebar.button("🚀 Entraîner / Ré-entraîner", use_container_width=True)

if bouton_train:
    with st.spinner(f"Téléchargement des données et entraînement pour {ville}... "
                    "(2 à 4 min à la première exécution)"):
        try:
            train_module.entrainer_pour_ville(
                ville, annees_test=annees_test, debut="2010-01-01", verbose=False,
                lat=lat, lon=lon,
            )
            st.cache_data.clear()
            st.sidebar.success(f"✅ Modèles entraînés pour {ville} !")
            st.rerun()
        except Exception as exc:  # noqa: BLE001 — affichage d'erreur utilisateur
            st.sidebar.error(f"❌ Échec de l'entraînement : {exc}")

st.sidebar.markdown("### 🤖 Assistant IA Tounsi")
mode_ia = st.sidebar.radio(
    "Moteur IA",
    ["Local Derja (Défaut)", "Ollama (DeepSeek-R1 7B)", "OpenAI GPT-4o-mini"],
    index=0,
    help="Choisissez le moteur d'intelligence artificielle pour l'assistant météo."
)

cle_openai_custom = ""
ollama_model_choice = "deepseek-r1:7b"
if mode_ia == "OpenAI GPT-4o-mini":
    cle_openai_custom = st.sidebar.text_input(
        "Clé OpenAI",
        type="password",
        help="Entrez votre clé API OpenAI.",
    )
elif mode_ia == "Ollama (DeepSeek-R1 7B)":
    ollama_model_choice = st.sidebar.text_input(
        "Modèle Ollama",
        value="deepseek-r1:7b",
        help="Nom du modèle local tournant sur Ollama (ex: deepseek-r1:7b).",
    )

st.sidebar.markdown("---")
st.sidebar.caption(
    "🇹🇳 Plateforme d'IA météo tunisienne — 24 Gouvernorats.\n"
    "Modèles : RandomForest / Gradient Boosting / XGBoost.\n"
    "NLP : Dialecte Tunisien (Derja & Arabizi) · Données : Open-Meteo & Meteostat."
)


# --------------------------------------------------------------------------
# Corps principal : Hero Banner & Carte
# --------------------------------------------------------------------------
modeles_ok = predict_module.modeles_complets(ville)

st.markdown(f"""
<div class="weather-hero">
    <div class="weather-hero-title">
        <span>🌤️ Plateforme Météo & IA Tunisie</span>
    </div>
    <p style="font-size: 1.05rem; opacity: 0.92; margin-bottom: 10px;">
        Intelligence météorologique par Machine Learning, données Open-Meteo & Assistant IA en <b>Dialecte Tunisien (Derja 🇹🇳)</b>.
    </p>
    <div style="display: flex; flex-wrap: wrap; gap: 8px;">
        <span class="weather-badge badge-active">📍 {ville} ({info_geo.get('nom_ar', '')}) · {lat:.2f}°N, {lon:.2f}°E</span>
        <span class="weather-badge badge-info">🧭 {info_geo.get('region')} ({info_geo.get('zone')})</span>
        <span class="weather-badge badge-info">🏞️ {info_geo.get('milieu')}</span>
        <span class="weather-badge {'badge-active' if info_geo.get('facade_maritime') else 'badge-warning'}">
            {'🌊 Façade Maritime' if info_geo.get('facade_maritime') else '🏞️ Région Intérieure'}
        </span>
        <span class="weather-badge {'badge-active' if modeles_ok else 'badge-warning'}">
            {'✅ 6 Modèles ML Prêts' if modeles_ok else '⏳ Modèles ML à entraîner (Live actif)'}
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

if mode_localisation == "Carte interactive":
    st.subheader("🗺️ Carte interactive des 24 Gouvernorats")
    carte = folium.Map(location=[34.6, 9.4], zoom_start=6, tiles="OpenStreetMap")
    for nom, (la, lo) in VILLES.items():
        pret = predict_module.modeles_complets(nom)
        couleur = "green" if pret else "blue"
        icone = "ok-sign" if pret else "info-sign"
        statut = "Modèle prêt ✅" if pret else "À entraîner ⏳"
        geo_item = obtenir_info_geographique(nom)
        popup_html = f"""
        <div style='font-family: sans-serif; min-width: 200px; font-size: 13px;'>
            <b style='font-size: 14px; color: #0284c7;'>📍 {nom} ({geo_item.get('nom_ar', '')})</b><br>
            <b>Région :</b> {geo_item.get('region')} ({geo_item.get('zone')})<br>
            <b>Milieu :</b> {geo_item.get('milieu')}<br>
            <b>Altitude :</b> ~{geo_item.get('altitude_m')} m<br>
            <b>Côte :</b> {'🌊 Littoral' if geo_item.get('facade_maritime') else '🏞️ Intérieure'}<br>
            <b>Voisins :</b> {', '.join(geo_item.get('voisins', []))}<br>
            <b>Statut ML :</b> {statut}<br>
            <p style='font-size: 11px; color: #475569; margin: 6px 0 0 0;'>{geo_item.get('description_fr')}</p>
        </div>
        """
        folium.Marker(
            [la, lo],
            tooltip=f"{nom} ({geo_item.get('region')}) — {statut}",
            popup=folium.Popup(popup_html, max_width=280),
            icon=folium.Icon(color=couleur, icon=icone),
        ).add_to(carte)
    if st.session_state.get("carte_lat") is not None:
        folium.Marker(
            [st.session_state["carte_lat"], st.session_state["carte_lon"]],
            tooltip="Point sélectionné",
            icon=folium.Icon(color="red", icon="star"),
        ).add_to(carte)

    etat_carte = st_folium(carte, height=420, use_container_width=True, key="carte_selection")
    if etat_carte and etat_carte.get("last_clicked"):
        lat = float(etat_carte["last_clicked"]["lat"])
        lon = float(etat_carte["last_clicked"]["lng"])
        st.session_state["carte_lat"] = lat
        st.session_state["carte_lon"] = lon
        ville = f"GPS_{lat:.2f}_{lon:.2f}"
    st.caption(f"📍 Point sélectionné : **{lat:.4f}°N, {lon:.4f}°E** — les prévisions ci-dessous s'y rapportent.")

# --------------------------------------------------------------------------
# Chargement des prévisions (ML si disponibles + Live Open-Meteo)
# --------------------------------------------------------------------------
resultat = None
if modeles_ok:
    try:
        with st.spinner("Calcul des prévisions Machine Learning…"):
            resultat = predict_module.predire_multijours(ville=ville, lat=lat, lon=lon, jours=horizon)
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Note prévisions ML : {exc}")
        resultat = None

try:
    prev_api = weather_api.prevision_journaliere(lat, lon, jours=horizon)
except Exception:  # noqa: BLE001
    prev_api = pd.DataFrame()

try:
    live = weather_api.conditions_actuelles(lat, lon)
except Exception:  # noqa: BLE001
    live = None

try:
    horaire = weather_api.prevision_horaire(lat, lon, jours=2)
except Exception:  # noqa: BLE001
    horaire = pd.DataFrame()

previsions = resultat["previsions"] if resultat else []
premier = previsions[0] if previsions else None
metriques = resultat["metriques"] if resultat else {}


def _api_du_jour(date_str: str):
    """Ligne de prévision brute Open-Meteo (UV, direction du vent) pour une date."""
    if prev_api.empty:
        return None
    horodatage = pd.to_datetime(date_str)
    return prev_api.loc[horodatage] if horodatage in prev_api.index else None


# --------------------------------------------------------------------------
# Onglets Principaux (Dashboard, Chatbot Tounsi, Modèles, Climatologie)
# --------------------------------------------------------------------------
onglet_prev, onglet_chat, onglet_perf, onglet_clim = st.tabs(
    ["🔮 Prévisions & Dashboard", "💬 Chatbot IA Tounsi 🇹🇳", "📊 Modèles ML & XAI", "🌦️ Climatologie"]
)

with onglet_prev:
    # 1. Conditions météo en direct (Hero Card)
    if live:
        temp_reel = live.get("temp_c")
        rhum_reel = live.get("rhum")
        wspd_reel = live.get("wspd")
        code_wmo = live.get("code")
        desc_ciel = live.get("description", "Temps variable")
        icone_wmo, desc_ciel_wmo = _icone_meteo(code_wmo)
        if desc_ciel == "Temps variable":
            desc_ciel = desc_ciel_wmo
        uv_val = live.get("uv")
        pres_val = live.get("pres")
        wdir_val = live.get("direction", "n/c")
        horodatage = live.get("horodatage", "Direct")

        ressenti_val = _ressenti_temperature(temp_reel, rhum_reel, wspd_reel)
        ressenti_str = (
            f"· Ressenti {U.fmt_temperature(ressenti_val, systeme)}"
            if ressenti_val is not None else ""
        )
        _uv_label, _uv_color, _uv_bg = _badge_uv(uv_val)
        badge_uv_html = (
            f'<span style="background:{_uv_bg}; color:{_uv_color}; border:1px solid {_uv_color}44; '
            f'border-radius:9999px; padding:5px 14px; font-size:0.82rem; font-weight:700; display:inline-block;">'
            f'☀️ UV : {_uv_label}</span>'
        )

        st.markdown(f"""
        <div class="live-hero-card">
            <div class="live-hero-header">
                <div>
                    <span class="live-pill"><span class="live-dot"></span> LIVE MÉTÉO EN DIRECT</span>
                    <h2 style="margin:8px 0 2px 0; font-size:1.85rem; font-weight:800; letter-spacing:-0.5px;">
                        {ville} <span style="font-size:1.25rem; opacity:0.8; font-weight:400;">({info_geo.get('nom_ar', '')})</span>
                    </h2>
                    <div style="font-size:0.92rem; opacity:0.85;">
                        {info_geo.get('region')} · {info_geo.get('zone')} · Relevé à {horodatage}
                    </div>
                </div>
                <div style="text-align:right;">
                    {badge_uv_html}
                </div>
            </div>
            <div class="live-hero-body">
                <div style="display:flex; align-items:center; gap:16px;">
                    <span style="font-size:3.5rem; line-height:1;">{icone_wmo}</span>
                    <div>
                        <div class="live-temp-display">
                            {U.fmt_temperature(temp_reel, systeme)}
                        </div>
                        <div style="font-size:0.95rem; font-weight:600; opacity:0.9;">
                            {desc_ciel} <span style="opacity:0.75; font-size:0.85rem; font-weight:400;">{ressenti_str}</span>
                        </div>
                    </div>
                </div>
                <div class="live-hero-stats">
                    <div class="live-stat-chip">
                        <span style="opacity:0.75;">💨 Vent :</span>
                        <b>{U.fmt_vitesse(wspd_reel, systeme)}</b>
                        <span style="opacity:0.75;">({wdir_val})</span>
                    </div>
                    <div class="live-stat-chip">
                        <span style="opacity:0.75;">💧 Humidité :</span>
                        <b>{_texte(rhum_reel, '{:.0f}', ' %')}</b>
                    </div>
                    <div class="live-stat-chip">
                        <span style="opacity:0.75;">⏲️ Pression :</span>
                        <b>{_texte(pres_val, '{:.0f}', ' hPa')}</b>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Fiche géographique & identité territoriale
    with st.expander(f"🗺️ Situation géographique & profil territorial : {ville} ({info_geo.get('nom_ar', '')})", expanded=False):
        gcol1, gcol2 = st.columns([1, 2])
        with gcol1:
            st.markdown(f"""
            - **Région :** `{info_geo.get('region')}` ({info_geo.get('zone')})
            - **Milieu / Relief :** {info_geo.get('milieu')}
            - **Altitude :** ~`{info_geo.get('altitude_m')} m`
            - **Façade maritime :** `{'🌊 Oui (Littoral)' if info_geo.get('facade_maritime') else '🏞️ Non (Gouvernorat intérieur)'}`
            - **Voisins limitrophes :** {', '.join(info_geo.get('voisins', [])) if info_geo.get('voisins') else 'Point GPS'}
            """)
        with gcol2:
            st.markdown(f"""
            **Situation géographique :**
            {info_geo.get('description_fr')}

            **Climat local & influence du relief :**
            *{info_geo.get('climat')}*
            """)

    st.markdown("---")

    if not (resultat and premier):
        st.info(
            f"💡 **Modèles ML non encore entraînés pour {ville} :** "
            f"Les prévisions multi-jours avancées, la jauge de risque de pluie et l'intervalle de confiance "
            f"s'activeront dès l'entraînement terminé (cliquez sur **🚀 Entraîner / Ré-entraîner** dans la barre latérale).\n\n"
            f"👉 Vous pouvez dès maintenant explorer les **conditions réelles ci-dessus** et échanger avec l'**Assistant IA Tounsi 🇹🇳** dans le 2ème onglet !"
        )
        date_demain = str(date.today() + timedelta(days=1))
        horaire_demain = weather_api.heure_pleine_demain(horaire, date_demain)
        if not horaire_demain.empty:
            st.subheader(f"⏱️ Prévision heure par heure (Open-Meteo) — {date_demain}")
            libelle_t = U.libelle_temperature(systeme)
            fig_heure = go.Figure()
            fig_heure.add_trace(go.Bar(
                x=horaire_demain.index, y=horaire_demain["prcp_prob"],
                name="Probabilité de pluie (%)", marker_color="#38bdf8", opacity=0.45,
                yaxis="y2",
            ))
            fig_heure.add_trace(go.Scatter(
                x=horaire_demain.index,
                y=U.serie_temperature(horaire_demain["temp"].tolist(), systeme),
                name=f"Température ({libelle_t})", mode="lines+markers",
                line=dict(color="#f43f5e", width=2.4),
            ))
            fig_heure.update_layout(
                height=340, margin=dict(l=20, r=20, t=20, b=20),
                yaxis=dict(title=f"Température ({libelle_t})"),
                yaxis2=dict(title="Pluie (%)", overlaying="y", side="right", range=[0, 100]),
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                hovermode="x unified",
            )
            fig_heure.update_xaxes(tickformat="%H:%M")
            _styliser_figure(fig_heure)
            st.plotly_chart(fig_heure, use_container_width=True)
    else:
        st.markdown(f"""
        <div style="display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:10px; margin-bottom:14px;">
            <div style="display:flex; align-items:center; gap:8px;">
                <span class="badge-active">🔮 Modèles ML</span>
                <span style="font-weight:700; font-size:1.1rem;">Prévisions sur {horizon} jour(s)</span>
            </div>
            <span style="font-size:0.85rem; opacity:0.8;">
                Dernier relevé observé : <b>{resultat['date_derniere_observation']}</b>
            </span>
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)
        delta_tmax = premier["ecart_moyenne"]
        col1.metric(
            "🌡️ Tmax demain",
            U.fmt_temperature(premier["tmax"], systeme),
            delta=(f"{U.fmt_delta_temperature(delta_tmax, systeme)} vs normale"
                   if delta_tmax is not None else None),
        )
        col2.metric(
            "🌧️ Risque de pluie demain",
            f"{premier['proba_pluie'] * 100:.0f} %",
            delta="Pluie probable ⛈️" if premier["risque_pluie"] else "Peu probable ☀️",
        )
        col3.metric("🌡️ Tmin demain", U.fmt_temperature(premier["tmin"], systeme))

        api_demain = _api_du_jour(premier["date"])
        col4.metric(
            "💨 Vent max demain",
            U.fmt_vitesse(premier["wspd"], systeme),
            delta=(f"🧭 {api_demain['direction']}" if api_demain is not None else None),
        )

        if premier["ic_bas"] is not None:
            st.caption(
                f"🎯 **Incertitude (IC 90 %)** sur la Tmax de demain : "
                f"{U.fmt_temperature(premier['ic_bas'], systeme)} → "
                f"{U.fmt_temperature(premier['ic_haut'], systeme)} — "
                "estimée par quantiles résiduels du jeu de test (élargie en √horizon)."
            )

        # Grille visuelle de prévisions multi-jours (Cards deck)
        st.markdown(f"#### 📅 Vue synthétique ({horizon} jours)")
        cards_html = []
        jours_semaine = ["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"]
        for p in previsions:
            p_dt = pd.to_datetime(p["date"])
            jour_nom = jours_semaine[p_dt.weekday()]
            jour_str = f"{jour_nom} {p_dt.day:02d}/{p_dt.month:02d}"
            is_demain = p["horizon"] == 1
            card_cls = "forecast-card card-today" if is_demain else "forecast-card"
            label_h = "Demain (J+1)" if is_demain else f"J+{p['horizon']}"

            # Emojis adaptés
            if p["risque_pluie"] or p["proba_pluie"] >= 0.5:
                ico = "🌧️"
            elif p["proba_pluie"] >= 0.25:
                ico = "⛅"
            else:
                ico = "☀️"

            tmax_str = U.fmt_temperature(p["tmax"], systeme)
            tmin_str = U.fmt_temperature(p["tmin"], systeme)
            rain_pct = int(round(p["proba_pluie"] * 100))
            rain_badge = f'<span style="background:rgba(56,189,248,0.18); color:#38bdf8; padding:2px 8px; border-radius:12px; font-size:0.75rem; font-weight:700;">💧 {rain_pct}%</span>'
            wspd_str = U.fmt_vitesse(p["wspd"], systeme)

            cards_html.append(f"""
            <div class="{card_cls}">
                <div style="font-size:0.75rem; font-weight:700; opacity:0.8; text-transform:uppercase;">{label_h}</div>
                <div style="font-size:0.85rem; font-weight:600; margin-bottom:4px;">{jour_str}</div>
                <div style="font-size:2.2rem; margin:6px 0;">{ico}</div>
                <div style="font-weight:800; font-size:1.15rem; color:#f87171;">{tmax_str}</div>
                <div style="font-size:0.9rem; opacity:0.75; margin-bottom:8px;">{tmin_str}</div>
                <div style="display:flex; flex-direction:column; gap:4px; align-items:center;">
                    {rain_badge}
                    <span style="font-size:0.75rem; opacity:0.8;">💨 {wspd_str}</span>
                </div>
            </div>
            """)

        st.markdown(f'<div class="forecast-grid">{"".join(cards_html)}</div>', unsafe_allow_html=True)

        col_gauche, col_droite = st.columns([1, 2])
        with col_gauche:
            st.subheader("☔ Probabilité de pluie demain")
            proba_pct = premier["proba_pluie"] * 100
            fig_jauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=proba_pct,
                number={"suffix": " %", "font": {"size": 34}},
                title={"text": "Risque de pluie", "font": {"size": 14}},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": _couleur_pluie(proba_pct), "thickness": 0.35},
                    "steps": [
                        {"range": [0, 25], "color": "rgba(46, 204, 113, 0.25)"},
                        {"range": [25, 50], "color": "rgba(241, 196, 15, 0.25)"},
                        {"range": [50, 75], "color": "rgba(230, 126, 34, 0.25)"},
                        {"range": [75, 100], "color": "rgba(231, 76, 60, 0.35)"},
                    ],
                    "threshold": {"line": {"color": "#ef4444", "width": 3}, "thickness": 0.9, "value": 50},
                },
            ))
            fig_jauge.update_layout(height=300, margin=dict(l=30, r=30, t=50, b=10))
            _styliser_figure(fig_jauge)
            st.plotly_chart(fig_jauge, use_container_width=True)
            st.caption(
                f"Seuil « jour pluvieux » : prcp > {resultat['seuil_pluie_mm']} mm — "
                f"F1 (test) = {metriques['rain'].get('f1', float('nan')):.2f} · "
                f"AUC = {metriques['rain'].get('auc', float('nan')):.2f}"
            )

        with col_droite:
            st.subheader(f"⏱️ Heure par heure — {premier['date']}")
            horaire_demain = weather_api.heure_pleine_demain(horaire, premier["date"])
            if horaire_demain.empty:
                st.info("Prévision horaire indisponible (API Open-Meteo injoignable).")
            else:
                libelle_t = U.libelle_temperature(systeme)
                fig_heure = go.Figure()
                fig_heure.add_trace(go.Bar(
                    x=horaire_demain.index, y=horaire_demain["prcp_prob"],
                    name="Probabilité de pluie (%)", marker_color="#38bdf8", opacity=0.45,
                    yaxis="y2",
                ))
                fig_heure.add_trace(go.Scatter(
                    x=horaire_demain.index,
                    y=U.serie_temperature(horaire_demain["temp"].tolist(), systeme),
                    name=f"Température ({libelle_t})", mode="lines+markers",
                    line=dict(color="#f43f5e", width=2.4),
                ))
                fig_heure.update_layout(
                    height=340, margin=dict(l=20, r=20, t=20, b=20),
                    yaxis=dict(title=f"Température ({libelle_t})"),
                    yaxis2=dict(title="Pluie (%)", overlaying="y", side="right", range=[0, 100]),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                    hovermode="x unified",
                )
                fig_heure.update_xaxes(tickformat="%H:%M")
                _styliser_figure(fig_heure)
                st.plotly_chart(fig_heure, use_container_width=True)

        st.markdown("---")
        st.subheader(f"📋 Prévisions détaillées ({horizon} jour(s))")

        lignes = []
        for p in previsions:
            api_jour = _api_du_jour(p["date"])
            ic = (f"{U.fmt_temperature(p['ic_bas'], systeme)} → {U.fmt_temperature(p['ic_haut'], systeme)}"
                  if p["ic_bas"] is not None else "n/c")
            lignes.append({
                "Date": p["date"],
                "Horizon": f"J+{p['horizon']}",
                f"Tmax ({U.libelle_temperature(systeme)})": U.fmt_temperature(p["tmax"], systeme),
                "IC 90 %": ic,
                f"Tmin ({U.libelle_temperature(systeme)})": U.fmt_temperature(p["tmin"], systeme),
                "Pluie": f"{p['proba_pluie'] * 100:.0f} %",
                f"Vent ({U.libelle_vitesse(systeme)})": U.fmt_vitesse(p["wspd"], systeme),
                "Humidité": f"{p['rhum']:.0f} %",
                "Pression (hPa)": f"{p['pres']:.0f}",
                "UV max": _texte(api_jour["uv"] if api_jour is not None else None, "{:.0f}"),
                "Direction vent": (api_jour["direction"] if api_jour is not None else "n/c"),
            })
        st.dataframe(pd.DataFrame(lignes), use_container_width=True, hide_index=True)
        st.caption(
            "Les colonnes **UV max** et **Direction vent** proviennent de la prévision brute "
            "Open-Meteo (variables non modélisées) ; le reste est produit par les modèles ML."
        )

        if horizon > 1:
            st.subheader("📈 Évolution des températures prévues")
            dates_trace = [pd.to_datetime(p["date"]) for p in previsions]
            tmax_trace = U.serie_temperature([p["tmax"] for p in previsions], systeme)
            tmin_trace = U.serie_temperature([p["tmin"] for p in previsions], systeme)
            libelle_t = U.libelle_temperature(systeme)

            fig_multi = go.Figure()
            if all(p["ic_bas"] is not None for p in previsions):
                haut = U.serie_temperature([p["ic_haut"] for p in previsions], systeme)
                bas = U.serie_temperature([p["ic_bas"] for p in previsions], systeme)
                fig_multi.add_trace(go.Scatter(
                    x=dates_trace, y=haut, mode="lines", line=dict(width=0),
                    showlegend=False, hoverinfo="skip",
                ))
                fig_multi.add_trace(go.Scatter(
                    x=dates_trace, y=bas, mode="lines", line=dict(width=0),
                    fill="tonexty", fillcolor="rgba(255,165,0,0.20)",
                    name="IC 90 %", hoverinfo="skip",
                ))
            fig_multi.add_trace(go.Scatter(
                x=dates_trace, y=tmax_trace, mode="lines+markers", name=f"Tmax ({libelle_t})",
                line=dict(color="#EF553B", width=2.4),
            ))
            fig_multi.add_trace(go.Scatter(
                x=dates_trace, y=tmin_trace, mode="lines+markers", name=f"Tmin ({libelle_t})",
                line=dict(color="#636EFA", width=2.4, dash="dot"),
            ))
            fig_multi.update_layout(
                height=380, margin=dict(l=20, r=20, t=30, b=20),
                yaxis_title=f"Température ({libelle_t})", xaxis_title="Date",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                hovermode="x unified",
            )
            _styliser_figure(fig_multi)
            st.plotly_chart(fig_multi, use_container_width=True)

# ==========================================================================
# Onglet 2 — Chatbot IA Tounsi (100% Dialecte Tunisien / Derja)
# ==========================================================================
with onglet_chat:
    st.markdown(f"""
    <div class="chat-welcome-card">
        <div style="display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:12px;">
            <div style="display:flex; align-items:center; gap:14px;">
                <span style="font-size:2.4rem;">🇹🇳</span>
                <div>
                    <h3 style="margin:0; font-size:1.35rem; color:#e11d48; font-weight:800;">
                        مساعد الطقس الذكي التونسي — Tounsi Weather AI
                    </h3>
                    <p style="margin:2px 0 0 0; font-size:0.95rem; opacity:0.88;">
                        Agent IA météorologique qui comprend à 100% le dialecte tunisien (Derja & Arabizi) et répond avec authenticité et précision.
                    </p>
                </div>
            </div>
            <div>
                <span class="badge-active">🇹🇳 100% Derja Tunisienne</span>
                <span class="badge-info">📍 {ville}</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("##### 💡 As'el b clique wa7da (Suggestions rapides) :")
    quick_tabs = st.tabs(["🌤️ Ta9s & Mtar", "👕 5rouj & Lbsa", "🌊 Ba7r & Joghrafya"])

    QUICK_PROMPTS = {
        0: [  # Ta9s & Mtar
            f"Chnowa l'jaw lyoum f {ville} ?",
            f"Fama mtar ghodwa f {ville} ?",
            f"Chnowa el 7arara ghodwa f {ville} ?",
            f"El jaw behi el jom3a ljaya f {ville} ?",
        ],
        1: [  # 5rouj & Lbsa
            f"Chnowa nelbes ghodwa sbah f {ville} ?",
            f"Lézem nekhdh parapluie ghodwa ?",
            f"Ta9s behi lil 5rouj w 9ahwa f {ville} ?",
            f"Fama ri7 barcha ghodwa ?",
        ],
        2: [  # Ba7r & Joghrafya
            f"El ba7r behi lil 3wema lyoum f {ville} ?",
            f"Win jéya {ville} géographiquement ? 🗺️",
            f"Chnowa el farq bin {ville} w Tunis f ta9s ?",
            f"Amel comparaison bin les gouvernorats tounsia ?",
        ],
    }

    for tab_idx, tab_obj in enumerate(quick_tabs):
        with tab_obj:
            btn_cols = st.columns(2)
            for btn_i, q in enumerate(QUICK_PROMPTS[tab_idx]):
                if btn_cols[btn_i % 2].button(q, key=f"btn_quick_{tab_idx}_{btn_i}", use_container_width=True):
                    st.session_state["pending_chat_prompt"] = q
                    st.rerun()

    # Initialisation de l'historique
    if "messages_tounsi" not in st.session_state:
        st.session_state["messages_tounsi"] = [
            {
                "role": "assistant",
                "content": (
                    f"**3aslema bik ya sidi !** Mar7ba bik 🇹🇳.\n\n"
                    f"Ena el assistant météo mte3ek b tounsi. Taw a7na f **{ville}** ({info_geo.get('region')}).\n"
                    f"As'elni 3la ay 7aja t7ebha : *win jéya {ville} géographiquement, el ta9s, el mtar, chnowa telbes, el ba7r wel 3wema, wela ay gouvernorat a5er f tounes !*"
                ),
            }
        ]

    st.markdown("---")

    # Affichage des messages existants
    for msg in st.session_state["messages_tounsi"]:
        avatar_icon = "🇹🇳" if msg["role"] == "assistant" else "👤"
        with st.chat_message(msg["role"], avatar=avatar_icon):
            st.markdown(msg["content"])

    # Zone de saisie
    prompt_user = st.chat_input("As'elni 3la ta9s b tounsi... (ex: fama mtar ghodwa f Sousse ?)")
    if not prompt_user and st.session_state.get("pending_chat_prompt"):
        prompt_user = st.session_state.pop("pending_chat_prompt")

    if prompt_user:
        st.session_state["messages_tounsi"].append({"role": "user", "content": prompt_user})
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt_user)

        with st.chat_message("assistant", avatar="🇹🇳"):
            with st.spinner("L'assistant 9a3ed yjeweb fik... 🇹🇳"):
                use_ollama_flag = (mode_ia == "Ollama (DeepSeek-R1 7B)") if "mode_ia" in locals() else False
                reponse_ai = tounsi_ai.generer_reponse_ai_tounsi(
                    prompt_user,
                    historique=st.session_state["messages_tounsi"][:-1],
                    gouvernorat_actif=ville,
                    api_key=cle_openai_custom if "cle_openai_custom" in locals() and cle_openai_custom else None,
                    use_ollama=use_ollama_flag,
                    ollama_model=ollama_model_choice if "ollama_model_choice" in locals() else "deepseek-r1:7b",
                )
                st.markdown(reponse_ai)
                st.session_state["messages_tounsi"].append({"role": "assistant", "content": reponse_ai})
        st.rerun()

    # Bouton de nettoyage
    if len(st.session_state["messages_tounsi"]) > 1:
        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        if st.button("🗑️ Fassakh el conversation (Effacer)", type="secondary"):
            st.session_state["messages_tounsi"] = [
                {
                    "role": "assistant",
                    "content": f"**Conversation tfasskhet !** 3aslema bik men jdid f **{ville}**. Chnowa t7eb ta3ref ?",
                }
            ]
            st.rerun()

# ==========================================================================
# Onglet 3 — Modèles & Explicabilité
# ==========================================================================
def _afficher_onglet_perf():
    st.subheader("🧾 Performances des modèles (jeu de test, split temporel)")
    lignes_perf = []
    for cible in CIBLES_REQUIS:
        met = metriques[cible]
        ligne = {"Cible": LIBELLES_CIBLES[cible], "Modèle": resultat["modeles"][cible]}
        if cible in CIBLES_REGRESSION:
            facteur = 1.0 if U.est_metric(systeme) else 1.8
            unite = f" ({U.libelle_temperature(systeme)})" if cible in ("tmax", "tmin") else ""
            ligne["MAE" + unite] = round(met["MAE"] * facteur, 3)
            ligne["RMSE" + unite] = round(met["RMSE"] * facteur, 3)
            ligne["R²"] = round(met["R2"], 4)
        else:
            ligne["Accuracy"] = round(met.get("accuracy", float("nan")), 3)
            ligne["Precision"] = round(met.get("precision", float("nan")), 3)
            ligne["Recall"] = round(met.get("recall", float("nan")), 3)
            ligne["F1"] = round(met.get("f1", float("nan")), 3)
            ligne["AUC"] = round(met.get("auc", float("nan")), 3)
        lignes_perf.append(ligne)
    st.dataframe(pd.DataFrame(lignes_perf), use_container_width=True, hide_index=True)

    infos = resultat.get("infos") or {}
    if infos:
        st.caption(
            f"Période d'entraînement : {infos['periode']['debut']} → {infos['periode']['fin']} — "
            f"test à partir du {infos['split']['test_debut']} "
            f"({infos['split']['annees_test']} ans). Entraîné le {infos.get('date_entrainement', 'n/c')}."
        )

    st.markdown("---")
    fichier_preds = os.path.join(DATA_DIR, f"test_preds_{ville}.csv")
    preds = pd.read_csv(fichier_preds, parse_dates=["date"], index_col="date") if os.path.exists(fichier_preds) else None

    col_perf1, col_perf2 = st.columns([3, 1])
    with col_perf1:
        st.subheader("📈 Température maximale : réel vs prédit (test)")
        if preds is not None:
            libelle_t = U.libelle_temperature(systeme)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=preds.index, y=U.serie_temperature(preds["tmax_reel"].tolist(), systeme),
                name="Réel", mode="lines", line=dict(color="#EF553B", width=2),
            ))
            fig.add_trace(go.Scatter(
                x=preds.index, y=U.serie_temperature(preds["tmax_pred"].tolist(), systeme),
                name="Prédit", mode="lines", line=dict(color="#00CC96", width=1.8, dash="dash"),
            ))
            fig.update_layout(
                height=400, margin=dict(l=20, r=20, t=30, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                xaxis_title="Date", yaxis_title=f"Tmax ({libelle_t})", hovermode="x unified",
            )
            fig.update_xaxes(rangeslider=dict(visible=True))
            _styliser_figure(fig)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Fichier de prédictions de test introuvable — ré-entraînez les modèles.")

    with col_perf2:
        st.subheader("🌧️ Risque de pluie (test)")
        if preds is not None:
            fig_pluie = go.Figure()
            fig_pluie.add_trace(go.Bar(
                x=preds.index, y=preds["rain_proba"] * 100,
                name="Probabilité prédite", marker_color="#636EFA", opacity=0.65,
            ))
            jours_pluie = preds[preds["rain_reel"] == 1]
            fig_pluie.add_trace(go.Scatter(
                x=jours_pluie.index, y=[105] * len(jours_pluie),
                name="Pluie observée", mode="markers",
                marker=dict(symbol="line-ns-open", size=9, color="#DC3912", line_width=2),
            ))
            fig_pluie.update_layout(
                height=400, margin=dict(l=10, r=10, t=30, b=20),
                yaxis=dict(range=[0, 115], title="Probabilité (%)"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                hovermode="x unified",
            )
            _styliser_figure(fig_pluie)
            st.plotly_chart(fig_pluie, use_container_width=True)

    st.markdown("---")
    st.subheader("🔍 Importances des variables (globale, par modèle)")
    cible_fi = st.selectbox(
        "Modèle à inspecter", list(CIBLES_REQUIS), index=0,
        format_func=lambda c: LIBELLES_CIBLES[c],
    )
    bundle_fi = predict_module.charger_modele(cible_fi, ville)
    importances = (
        pd.Series(bundle_fi["model"].feature_importances_, index=bundle_fi["features"])
        .sort_values(ascending=True).tail(12)
    )
    fig_imp = go.Figure(go.Bar(
        x=importances.values, y=importances.index, orientation="h",
        marker_color="#00CC96",
    ))
    fig_imp.update_layout(
        height=400, margin=dict(l=20, r=20, t=10, b=20),
        xaxis_title="Importance (réduction d'impureté)", yaxis_title="",
    )
    _styliser_figure(fig_imp)
    st.plotly_chart(fig_imp, use_container_width=True)

    st.markdown("---")
    st.subheader("🧠 Explicabilité dynamique (SHAP — XAI)")
    if not explain_module.shap_disponible():
        st.warning("La bibliothèque `shap` n'est pas installée — `pip install shap` pour activer cette section.")
    else:
        col_xai1, col_xai2 = st.columns([1, 2])
        with col_xai1:
            cible_xai = st.selectbox(
                "Cible expliquée", list(CIBLES_REQUIS), index=0,
                format_func=lambda c: LIBELLES_CIBLES[c], key="xai_cible",
            )
            source_xai = st.radio("Jour analysé", ["Demain (J+1)", "Un jour du jeu de test"], index=0)

            X_row = None
            libelle_jour = ""
            if source_xai == "Demain (J+1)":
                X_row = premier["features_row"]
                libelle_jour = premier["date"]
            elif preds is not None and len(preds):
                test_debut = preds.index.min().date()
                test_fin = preds.index.max().date()
                jour_choisi = st.date_input(
                    "Date (jeu de test)", value=test_fin, min_value=test_debut, max_value=test_fin,
                )
                libelle_jour = str(jour_choisi)
                try:
                    debut_hist = f"{str(infos['periode']['debut'])[:4]}-01-01" if infos else "2010-01-01"
                    features_hist = _features_historique(
                        lat, lon, debut_hist, (date.today() - timedelta(days=1)).isoformat()
                    )
                    horodatage = pd.to_datetime(jour_choisi)
                    if horodatage in features_hist.index:
                        X_row = features_hist.loc[[horodatage]]
                    else:
                        st.warning("Date absente de l'historique disponible (features indisponibles).")
                except Exception as exc:  # noqa: BLE001
                    st.warning(f"Historique indisponible pour l'explication : {exc}")

        with col_xai2:
            if X_row is None:
                st.info("Sélectionnez un jour valide pour afficher l'explication SHAP.")
            else:
                bundle_xai = predict_module.charger_modele(cible_xai, ville)
                explication = explain_module.explication_shap(bundle_xai, X_row)
                if explication is None:
                    st.warning("Explication SHAP indisponible pour ce modèle/jour.")
                else:
                    items = sorted(explication["valeurs"].items(), key=lambda kv: abs(kv[1]))[-12:]
                    noms = [k for k, _ in items]
                    valeurs = [v for _, v in items]
                    unite_xai = explication["unite"]
                    if unite_xai == "°C" and not U.est_metric(systeme):
                        valeurs = [U.delta_temperature(v, systeme) for v in valeurs]
                        unite_xai = "°F"

                    fig_shap = go.Figure(go.Bar(
                        x=valeurs, y=noms, orientation="h",
                        marker_color=["#EF553B" if v > 0 else "#636EFA" for v in valeurs],
                    ))
                    titre_axe = f"Contribution à la prédiction ({unite_xai})" if unite_xai else "Contribution (log-odds)"
                    fig_shap.update_layout(
                        height=420, margin=dict(l=20, r=20, t=30, b=20),
                        title=f"Explication SHAP — {LIBELLES_CIBLES[cible_xai]} — {libelle_jour}",
                        xaxis_title=titre_axe, yaxis_title="",
                    )
                    _styliser_figure(fig_shap)
                    st.plotly_chart(fig_shap, use_container_width=True)
                    base = explication["base"]
                    pred = explication["prediction"]
                    if unite_xai == "°C" or unite_xai == "°F":
                        st.caption(
                            f"Lecture : partant d'une valeur de base de "
                            f"{U.fmt_temperature(base, systeme)}, les variables ci-dessus poussent la "
                            f"prédiction vers **{U.fmt_temperature(pred, systeme)}** "
                            "(rouge = augmente, bleu = diminue)."
                        )
                    else:
                        st.caption(
                            f"Lecture : la probabilité de pluie prédite est **{pred * 100:.0f} %** "
                            "(contributions en échelle log-odds ; rouge augmente, bleu diminue)."
                        )

    st.markdown("---")
    st.caption(
        "Pipeline : Open-Meteo / Meteostat → pandas/numpy → feature engineering (lags, "
        "moyennes mobiles, calendrier cyclique) → RandomForest / Gradient Boosting / XGBoost "
        "→ intervalles de confiance (quantiles résiduels) → Streamlit + Plotly + Folium + SHAP."
    )


with onglet_perf:
    if not resultat:
        st.info(
            f"👋 Les modèles de Machine Learning de **{ville}** ne sont pas encore entraînés.\n\n"
            f"Cliquez sur **🚀 Entraîner / Ré-entraîner** dans la barre latérale pour entraîner les 6 modèles (2 à 3 min) et visualiser leurs performances et l'explicabilité SHAP."
        )
    else:
        _afficher_onglet_perf()

# ==========================================================================
# Onglet 4 — Climatologie
# ==========================================================================
with onglet_clim:
    st.subheader("🌦️ Climatologie locale : situer la prévision dans le temps long")
    infos = (resultat.get("infos") or {}) if resultat else {}
    debut_clim = f"{str(infos.get('periode', {}).get('debut', '2010-01-01'))[:4]}-01-01"
    try:
        df_hist = _charger_climatologie(lat, lon, debut_clim)
    except Exception as exc:  # noqa: BLE001
        df_hist = pd.DataFrame()
        st.error(f"Historique climatologique indisponible : {exc}")

    if df_hist is None or len(df_hist) < 365:
        st.warning("Pas assez d'historique pour l'analyse climatologique (≥ 1 an requis).")
    else:
        normales = climatology.normales_mensuelles(df_hist)
        libelle_t = U.libelle_temperature(systeme)

        col_clim1, col_clim2 = st.columns([3, 2])
        with col_clim1:
            fig_norm = go.Figure()
            fig_norm.add_trace(go.Scatter(
                x=MOIS_ORDRE, y=U.serie_temperature(normales["tmax"].tolist(), systeme),
                name=f"Normale Tmax ({libelle_t})", mode="lines+markers",
                line=dict(color="#00CC96", width=2.4),
            ))
            fig_norm.add_trace(go.Scatter(
                x=MOIS_ORDRE, y=U.serie_temperature(normales["tmin"].tolist(), systeme),
                name=f"Normale Tmin ({libelle_t})", mode="lines+markers",
                line=dict(color="#636EFA", width=2.0, dash="dot"),
            ))
            fig_norm.add_trace(go.Scatter(
                x=[climatology.MOIS_FR[pd.to_datetime(p["date"]).month] for p in previsions],
                y=U.serie_temperature([p["tmax"] for p in previsions], systeme),
                name="Prévision Tmax", mode="markers+text",
                text=[f"J+{p['horizon']}" for p in previsions],
                textposition="top center",
                marker=dict(color="#EF553B", size=12, symbol="diamond"),
            ))
            fig_norm.update_layout(
                height=400, margin=dict(l=20, r=20, t=30, b=20),
                title="Normales mensuelles vs prévision",
                xaxis_title="Mois", yaxis_title=f"Température ({libelle_t})",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                hovermode="x unified",
            )
            fig_norm.update_xaxes(categoryorder="array", categoryarray=MOIS_ORDRE)
            _styliser_figure(fig_norm)
            st.plotly_chart(fig_norm, use_container_width=True)

        with col_clim2:
            fig_pluie_mois = go.Figure(go.Bar(
                x=MOIS_ORDRE, y=normales["prcp"].tolist(),
                marker_color="#636EFA", name="Cumul moyen",
            ))
            fig_pluie_mois.update_layout(
                height=400, margin=dict(l=20, r=20, t=30, b=20),
                title="Pluie cumulée moyenne par mois",
                xaxis_title="Mois", yaxis_title="Précipitations (mm)",
            )
            fig_pluie_mois.update_xaxes(categoryorder="array", categoryarray=MOIS_ORDRE)
            _styliser_figure(fig_pluie_mois)
            st.plotly_chart(fig_pluie_mois, use_container_width=True)

        col_clim3, col_clim4 = st.columns(2)
        with col_clim3:
            df_mois = climatology.mois_dataframe(df_hist)
            fig_box = go.Figure(go.Box(
                x=df_mois["month_label"],
                y=U.serie_temperature(df_mois["tmax"].tolist(), systeme),
                marker_color="#FFA15A", boxpoints=False,
            ))
            fig_box.update_layout(
                height=380, margin=dict(l=20, r=20, t=30, b=20),
                title="Distribution de la Tmax par mois",
                xaxis_title="Mois", yaxis_title=f"Tmax ({libelle_t})",
            )
            fig_box.update_xaxes(categoryorder="array", categoryarray=MOIS_ORDRE)
            _styliser_figure(fig_box)
            st.plotly_chart(fig_box, use_container_width=True)

        with col_clim4:
            saisons = climatology.normales_saisonnières(df_hist)
            fig_saison = go.Figure()
            fig_saison.add_trace(go.Bar(
                x=saisons.index, y=U.serie_temperature(saisons["tmax"].tolist(), systeme),
                name=f"Tmax ({libelle_t})", marker_color="#EF553B",
            ))
            fig_saison.add_trace(go.Bar(
                x=saisons.index, y=U.serie_temperature(saisons["tmin"].tolist(), systeme),
                name=f"Tmin ({libelle_t})", marker_color="#636EFA",
            ))
            fig_saison.update_layout(
                height=380, margin=dict(l=20, r=20, t=30, b=20),
                title="Normales saisonnières", barmode="group",
                yaxis_title=f"Température ({libelle_t})",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            _styliser_figure(fig_saison)
            st.plotly_chart(fig_saison, use_container_width=True)

        st.subheader("🔥 Anomalies annuelles de température (réchauffement local)")
        anomalies = climatology.anomalies_annuelles(df_hist)
        fig_anom = go.Figure(go.Bar(
            x=[str(a) for a in anomalies.index],
            y=[U.delta_temperature(v, systeme) for v in anomalies["ecart"].tolist()],
            marker_color=["#EF553B" if v > 0 else "#636EFA" for v in anomalies["ecart"]],
        ))
        fig_anom.update_layout(
            height=360, margin=dict(l=20, r=20, t=30, b=20),
            xaxis_title="Année", yaxis_title=f"Écart de Tmax moyenne ({libelle_t})",
        )
        _styliser_figure(fig_anom)
        st.plotly_chart(fig_anom, use_container_width=True)
        st.caption(
            f"Référence : moyenne de la période {df_hist.index.min().date()} → "
            f"{df_hist.index.max().date()} ({len(df_hist)} jours). "
            "Barres rouges = année plus chaude que la normale, bleues = plus fraîche."
        )

# --------------------------------------------------------------------------
# Pied de page
# --------------------------------------------------------------------------
st.markdown("---")
st.caption(
    "Dashboards : carte interactive Folium · prévisions multi-jours & horaires · intervalles de "
    "confiance · SHAP (XAI) · analyse climatologique · données temps réel Open-Meteo."
)
