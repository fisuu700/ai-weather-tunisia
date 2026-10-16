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
from src.data_loader import VILLES             # noqa: E402

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
    margin-bottom: 24px;
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

/* Chat Welcome Card */
.chat-welcome-card {
    background: linear-gradient(135deg, rgba(225, 29, 72, 0.09) 0%, rgba(244, 63, 94, 0.05) 50%, rgba(251, 146, 60, 0.05) 100%);
    border: 1px solid rgba(244, 63, 94, 0.28);
    border-radius: 18px;
    padding: 22px 26px;
    margin-bottom: 22px;
    box-shadow: 0 8px 24px -6px rgba(225, 29, 72, 0.08);
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


# --------------------------------------------------------------------------
# Fonctions utilitaires (cache + formatage)
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
    liste_villes = list(VILLES.keys())
    # Sousse ou Tunis en premier par défaut car modèles déjà entraînés
    idx_defaut = liste_villes.index("Tunis") if "Tunis" in liste_villes else (
        liste_villes.index("Sousse") if "Sousse" in liste_villes else 0
    )
    ville = st.sidebar.selectbox(
        "Gouvernorat (24 en Tunisie)",
        liste_villes,
        index=idx_defaut,
        format_func=lambda v: f"{v} {'(Modèle prêt ✅)' if predict_module.modeles_complets(v) else '(À entraîner ⏳)'}",
    )
    lat, lon = VILLES[ville]
elif mode_localisation == "Coordonnées GPS":
    st.sidebar.caption("Ex. Tunis : 36.81 / 10.18 — Sousse : 35.83 / 10.64 — Sfax : 34.74 / 10.76")
    lat = st.sidebar.number_input("Latitude", -90.0, 90.0, 36.8065, step=0.0001, format="%.4f")
    lon = st.sidebar.number_input("Longitude", -180.0, 180.0, 10.1815, step=0.0001, format="%.4f")
    ville = f"GPS_{lat:.2f}_{lon:.2f}"
else:
    lat = st.session_state.get("carte_lat", VILLES["Tunis"][0])
    lon = st.session_state.get("carte_lon", VILLES["Tunis"][1])
    ville = f"GPS_{lat:.2f}_{lon:.2f}"
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
cle_openai_custom = st.sidebar.text_input(
    "Clé OpenAI (optionnelle)",
    type="password",
    help="Par défaut, le moteur local 100% tunisien (Derja) est actif sans clé. Vous pouvez renseigner votre clé OpenAI pour activer GPT-4o-mini.",
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
    <div>
        <span class="weather-badge badge-active">📍 {ville} ({lat:.2f}°N, {lon:.2f}°E)</span>
        <span class="weather-badge {'badge-active' if modeles_ok else 'badge-warning'}">
            {'✅ 6 Modèles ML Disponibles' if modeles_ok else '⏳ Modèles ML à entraîner (Live actif)'}
        </span>
        <span class="weather-badge badge-info">🇹🇳 Chatbot Tounsi 100% Connecté</span>
        <span class="weather-badge badge-info">🛰️ 24 Gouvernorats</span>
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
        folium.Marker(
            [la, lo],
            tooltip=f"{nom} ({statut})",
            popup=f"<b>{nom}</b><br>{statut}",
            icon=folium.Icon(color=couleur, icon=icone),
        ).add_to(carte)
    if st.session_state.get("carte_lat") is not None:
        folium.Marker(
            [st.session_state["carte_lat"], st.session_state["carte_lon"]],
            tooltip="Point sélectionné",
            icon=folium.Icon(color="red", icon="star"),
        ).add_to(carte)

    etat_carte = st_folium(carte, height=400, use_container_width=True, key="carte_selection")
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
    if not resultat:
        st.warning(
            f"ℹ️ **Modèles ML à entraîner pour {ville} :** "
            f"Cliquez sur **🚀 Entraîner / Ré-entraîner** dans la barre latérale pour entraîner les 6 modèles ML (2 à 3 min).\n\n"
            f"En attendant, retrouvez les **conditions temps réel ci-dessous** et discutez avec l'**Assistant IA Tounsi** dans le 2ème onglet !"
        )
    else:
        st.info(
            f"📅 Prévisions pour les **{horizon} prochain(s) jour(s)** — "
            f"à partir du dernier relevé observé : **{resultat['date_derniere_observation']}**."
        )

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

    if live:
        st.markdown("#### 🛰️ Conditions actuelles (Open-Meteo — temps réel)")
        lc = st.columns(6)
        lc[0].metric("Température", U.fmt_temperature(live.get("temp_c"), systeme))
        lc[1].metric("Ciel", live.get("description", "n/c"))
        lc[2].metric("Humidité", _texte(live.get("rhum"), "{:.0f}", " %"))
        lc[3].metric("Vent", U.fmt_vitesse(live.get("wspd"), systeme),
                     delta=f"🧭 {live.get('direction', 'n/c')}")
        lc[4].metric("Pression", _texte(live.get("pres"), "{:.0f}", " hPa"))
        lc[5].metric("Indice UV", _texte(live.get("uv"), "{:.1f}"))
        st.caption(f"Relevé à : {live.get('horodatage', 'n/c')} (heure locale du point).")

    st.markdown("---")

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
                    {"range": [0, 25], "color": "#D4EDDA"},
                    {"range": [25, 50], "color": "#FFF3CD"},
                    {"range": [50, 75], "color": "#F8D7DA"},
                    {"range": [75, 100], "color": "#F5C6CB"},
                ],
                "threshold": {"line": {"color": "black", "width": 3}, "thickness": 0.9, "value": 50},
            },
        ))
        fig_jauge.update_layout(height=300, margin=dict(l=30, r=30, t=50, b=10))
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
                name="Probabilité de pluie (%)", marker_color="#636EFA", opacity=0.35,
                yaxis="y2",
            ))
            fig_heure.add_trace(go.Scatter(
                x=horaire_demain.index,
                y=U.serie_temperature(horaire_demain["temp"].tolist(), systeme),
                name=f"Température ({libelle_t})", mode="lines+markers",
                line=dict(color="#EF553B", width=2),
            ))
            fig_heure.update_layout(
                height=340, margin=dict(l=20, r=20, t=20, b=20),
                yaxis=dict(title=f"Température ({libelle_t})"),
                yaxis2=dict(title="Pluie (%)", overlaying="y", side="right", range=[0, 100]),
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                hovermode="x unified",
            )
            fig_heure.update_xaxes(tickformat="%H:%M")
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
    c1, c2, c3 = st.columns(3)
    quick_list = [
        f"Chnowa l'jaw lyoum f {ville} ?",
        f"Fama mtar ghodwa f {ville} ?",
        f"Chnowa nelbes ghodwa sbah ?",
        f"El ba7r behi lil 3wema ?",
        f"Fama rya7 9wiya wela dhabeb ?",
        f"Ta9s behi lil 5rouj w 9ahwa ?",
    ]

    for i, q in enumerate(quick_list):
        col_btn = [c1, c2, c3][i % 3]
        if col_btn.button(q, key=f"btn_quick_{i}", use_container_width=True):
            st.session_state["pending_chat_prompt"] = q
            st.rerun()

    # Initialisation de l'historique
    if "messages_tounsi" not in st.session_state:
        st.session_state["messages_tounsi"] = [
            {
                "role": "assistant",
                "content": (
                    f"**3aslema bik ya sidi !** Mar7ba bik 🇹🇳.\n\n"
                    f"Ena el assistant météo mte3ek b tounsi. Taw a7na f **{ville}**.\n"
                    f"As'elni 3la ay 7aja t7ebha : *el ta9s, el mtar, chnowa telbes, el ba7r wel 3wema, wela ay gouvernorat a5er f tounes !*"
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
            with st.spinner("9a3ed nchouflek fel météo..."):
                reponse_ai = tounsi_ai.generer_reponse_ai_tounsi(
                    prompt_user,
                    historique=st.session_state["messages_tounsi"][:-1],
                    gouvernorat_actif=ville,
                    api_key=cle_openai_custom if "cle_openai_custom" in locals() and cle_openai_custom else None,
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
