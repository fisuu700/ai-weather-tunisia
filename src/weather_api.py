"""
weather_api.py
==============
Accès aux données MÉTÉOROLOGIQUES EN TEMPS RÉEL (API Open-Meteo, sans clé).

Ce module complète `data_loader` (historique) avec les flux « live » :

  1. `conditions_actuelles`  — relevés instantanés (température, vent,
     humidité, pression, UV, code météo traduit) ;
  2. `prevision_journaliere` — prévision brute du fournisseur pour les
     prochains jours : direction du vent dominante + indice UV max
     (deux variables que les modèles ML ne prédisent pas) ;
  3. `prevision_horaire`     — prévision heure par heure pour la
     journée à venir (température, probabilité de pluie, vent,
     humidité, UV, code météo).

Toutes les fonctions renvoient `None` / un DataFrame vide en cas
d'indisponibilité réseau : l'interface dégrade gracieusement sans jamais
planter (les prévisions ML restent affichées).
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
import warnings
from datetime import date, timedelta

import pandas as pd

URL_PREVISION = "https://api.open-meteo.com/v1/forecast"

# Codes météo WMO → libellés français (api.open-meteo.com, paramètre weather_code)
CODES_METEO: dict[int, str] = {
    0: "Ciel dégagé",
    1: "Principalement dégagé",
    2: "Partiellement nuageux",
    3: "Couvert",
    45: "Brouillard",
    48: "Brouillard givrant",
    51: "Bruine légère",
    53: "Bruine modérée",
    55: "Bruine dense",
    56: "Bruine verglaçante légère",
    57: "Bruine verglaçante dense",
    61: "Pluie faible",
    63: "Pluie modérée",
    65: "Pluie forte",
    66: "Pluie verglaçante légère",
    67: "Pluie verglaçante forte",
    71: "Neige faible",
    73: "Neige modérée",
    75: "Neige forte",
    77: "Grains de neige",
    80: "Averses faibles",
    81: "Averses modérées",
    82: "Averses violentes",
    85: "Averses de neige faibles",
    86: "Averses de neige fortes",
    95: "Orage",
    96: "Orage et grêle faible",
    99: "Orage et grêle forte",
}

ROUES_VENT: dict[int, str] = {
    0: "N", 1: "NNE", 2: "NE", 3: "ENE", 4: "E", 5: "ESE",
    6: "SE", 7: "SSE", 8: "S", 9: "SSO", 10: "SO", 11: "OSO",
    12: "O", 13: "ONO", 14: "NO", 15: "NNO",
}


def description_meteo(code: int | None) -> str:
    """Traduit un code météo WMO en libellé français lisible."""
    if code is None or pd.isna(code):
        return "n/c"
    return CODES_METEO.get(int(code), f"Code météo {int(code)}")


def rose_des_vents(degres: float | None) -> str:
    """Convertit une direction en degrés vers un point cardinal (16 roues)."""
    if degres is None or pd.isna(degres):
        return "n/c"
    return ROUES_VENT[int(round(float(degres) % 360 / 22.5)) % 16]


def _requete(params: dict) -> dict | None:
    """Interroge l'API Open-Meteo et décode le JSON, ou None si échec."""
    url = f"{URL_PREVISION}?{urllib.parse.urlencode(params)}"
    try:
        with urllib.request.urlopen(url, timeout=30) as reponse:
            return json.loads(reponse.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as exc:
        warnings.warn(f"API Open-Meteo inaccessible : {exc}")
        return None


# --------------------------------------------------------------------------
# 1. Conditions actuelles
# --------------------------------------------------------------------------
def conditions_actuelles(lat: float, lon: float) -> dict | None:
    """
    Relevés instantanés au point demandé (paramètre `current`).

    Retour : dict avec temp_c, rhum, prcp, code, description, wspd,
    wdir, direction, pres, uv, horodatage — ou None si indisponible.
    """
    contenu = _requete({
        "latitude": f"{lat:.4f}",
        "longitude": f"{lon:.4f}",
        "current": ",".join([
            "temperature_2m", "relative_humidity_2m", "precipitation",
            "weather_code", "wind_speed_10m", "wind_direction_10m",
            "pressure_msl", "uv_index",
        ]),
        "timezone": "auto",
    })
    if not contenu or "current" not in contenu:
        return None
    courant = contenu["current"]
    code = courant.get("weather_code")
    return {
        "horodatage": courant.get("time"),
        "temp_c": courant.get("temperature_2m"),
        "rhum": courant.get("relative_humidity_2m"),
        "prcp": courant.get("precipitation"),
        "code": code,
        "description": description_meteo(code),
        "wspd": courant.get("wind_speed_10m"),
        "wdir": courant.get("wind_direction_10m"),
        "direction": rose_des_vents(courant.get("wind_direction_10m")),
        "pres": courant.get("pressure_msl"),
        "uv": courant.get("uv_index"),
    }


# --------------------------------------------------------------------------
# 2. Prévision journalière (direction du vent + UV)
# --------------------------------------------------------------------------
def prevision_journaliere(lat: float, lon: float, jours: int = 7) -> pd.DataFrame:
    """
    Prévision brute Open-Meteo sur `jours` jours : direction du vent
    dominante et indice UV maximum (variables non modélisées par ML).

    Retour : DataFrame indexé par date (colonnes wdir, uv), possibly vide.
    """
    contenu = _requete({
        "latitude": f"{lat:.4f}",
        "longitude": f"{lon:.4f}",
        "forecast_days": max(1, min(int(jours), 16)),
        "daily": "wind_direction_10m_dominant,uv_index_max",
        "timezone": "auto",
    })
    if not contenu or not contenu.get("daily", {}).get("time"):
        return pd.DataFrame()
    quotidien = contenu["daily"]
    df = pd.DataFrame({
        "date": pd.to_datetime(quotidien["time"]),
        "wdir": quotidien.get("wind_direction_10m_dominant"),
        "uv": quotidien.get("uv_index_max"),
    }).set_index("date")
    df["direction"] = [rose_des_vents(v) for v in df["wdir"]]
    return df


# --------------------------------------------------------------------------
# 3. Prévision horaire
# --------------------------------------------------------------------------
def prevision_horaire(lat: float, lon: float, jours: int = 2) -> pd.DataFrame:
    """
    Prévision heure par heure (température, probabilité de pluie, vent,
    humidité, UV, code météo) pour les `jours` prochains jours locaux.

    Retour : DataFrame indexé par horodatage horaire.
    """
    horaires = ",".join([
        "temperature_2m", "relative_humidity_2m", "precipitation_probability",
        "precipitation", "wind_speed_10m", "wind_direction_10m",
        "uv_index", "weather_code",
    ])
    contenu = _requete({
        "latitude": f"{lat:.4f}",
        "longitude": f"{lon:.4f}",
        "hourly": horaires,
        "forecast_days": max(1, min(int(jours), 7)),
        "timezone": "auto",
    })
    if not contenu or not contenu.get("hourly", {}).get("time"):
        return pd.DataFrame()
    horaire = contenu["hourly"]
    df = pd.DataFrame({
        "time": pd.to_datetime(horaire["time"]),
        "temp": horaire.get("temperature_2m"),
        "rhum": horaire.get("relative_humidity_2m"),
        "prcp_prob": horaire.get("precipitation_probability"),
        "prcp": horaire.get("precipitation"),
        "wspd": horaire.get("wind_speed_10m"),
        "wdir": horaire.get("wind_direction_10m"),
        "uv": horaire.get("uv_index"),
        "code": horaire.get("weather_code"),
    }).set_index("time")
    return df


def heure_pleine_demain(horaire: pd.DataFrame, date_cible: date | str) -> pd.DataFrame:
    """Sous-ensemble horaire correspondant à la journée `date_cible`."""
    if horaire.empty:
        return horaire
    jour = pd.to_datetime(date_cible).normalize()
    masque = (horaire.index >= jour) & (horaire.index < jour + pd.Timedelta(days=1))
    return horaire.loc[masque]
