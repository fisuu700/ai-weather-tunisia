"""
data_loader.py
==============
Récupération et prétraitement des données météorologiques historiques.

Sources de données (deux backends interchangeables) :

  1. **Open-Meteo** (source primaire, défaut) — API REST gratuite et sans clé :
     - `archive-api.open-meteo.com`  : réanalyse ERA5, données jusqu'à hier ;
     - `api.open-meteo.com`          : modèle de prévision, complète les
       derniers jours (jusqu'à aujourd'hui) si l'archive ne les a pas encore.
     Variables : tmax, tmin, tavg, prcp, rhum, wspd, wdir, pres (journalier,
     agrégées par le fournisseur selon le fuseau local de la zone).

  2. **Meteostat** (repli optionnel, bibliothèque `meteostat`) — données
     horaires agrégées en journalier. Nécessite un accès réseau à
     images.meteostat.net ; si indisponible, le pipeline bascule sur Open-Meteo.

Mise en cache : chaque téléchargement est sauvegardé en CSV dans ./data
(un fichier par source, zone géographique et période) afin d'éviter de
re-télécharger le même historique à chaque exécution.

Sortie : un DataFrame indexé par date (DatetimeIndex, fréquence journalière)
contenant au minimum les colonnes :
    tavg, tmin, tmax, prcp, rhum, wspd, wdir, pres
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
import warnings
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

# 24 Gouvernorats de la Tunisie : nom -> (latitude, longitude)
VILLES: dict[str, tuple[float, float]] = {
    "Ariana": (36.8665, 10.1647),
    "Béja": (36.7256, 9.1817),
    "Ben Arous": (36.7531, 10.2189),
    "Bizerte": (37.2744, 9.8739),
    "Gabès": (33.8815, 10.0982),
    "Gafsa": (34.4250, 8.7842),
    "Jendouba": (36.5011, 8.7802),
    "Kairouan": (35.6781, 10.0963),
    "Kasserine": (35.1676, 8.8365),
    "Kébili": (33.7044, 8.9690),
    "Le Kef": (36.1822, 8.7149),
    "Mahdia": (35.5047, 11.0622),
    "Manouba": (36.8080, 10.0972),
    "Médenine": (33.3549, 10.5055),
    "Monastir": (35.7780, 10.8262),
    "Nabeul": (36.4561, 10.7376),
    "Sfax": (34.7406, 10.7603),
    "Sidi Bouzid": (35.0382, 9.4849),
    "Siliana": (36.0850, 9.3708),
    "Sousse": (35.8256, 10.6370),
    "Tataouine": (32.9297, 10.4518),
    "Tozeur": (33.9197, 8.1335),
    "Tunis": (36.8065, 10.1815),
    "Zaghouan": (36.4029, 10.1429),
}

import unicodedata


def normaliser_ville(nom: str) -> str:
    """Résout le nom officiel du gouvernorat (tolérant à la casse, espaces et accents)."""
    if nom in VILLES:
        return nom

    def _strip(txt: str) -> str:
        txt = unicodedata.normalize("NFD", txt)
        return "".join(c for c in txt if unicodedata.category(c) != "Mn").lower().strip()

    nom_clean = _strip(nom)
    # Recherche exacte normalisée
    for v in VILLES:
        if _strip(v) == nom_clean:
            return v
    # Tolérance pour "Kef" -> "Le Kef"
    if nom_clean in ("kef", "elkef", "el kef"):
        return "Le Kef"
    return nom


# Colonnes produites par le pipeline (ordre stable)
COLONNES_SORTIE = ["tavg", "tmin", "tmax", "prcp", "rhum", "wspd", "wdir", "pres"]

# ---- Open-Meteo : mapping variable API -> colonne interne ----
URL_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"   # ERA5, jusqu'à hier
URL_PREVISION = "https://api.open-meteo.com/v1/forecast"        # derniers jours (~92 j)
VARIABLES_OPEN_METEO = {
    "temperature_2m_max": "tmax",
    "temperature_2m_min": "tmin",
    "temperature_2m_mean": "tavg",
    "precipitation_sum": "prcp",
    "relative_humidity_2m_mean": "rhum",
    "wind_speed_10m_max": "wspd",          # vitesse max du vent à 10 m
    "wind_direction_10m_dominant": "wdir", # direction dominante
    "pressure_msl_mean": "pres",
}

# ---- Meteostat : agrégation horaire -> journalière ----
AGGREGATION_HORAIRE = {
    "tavg": "mean", "tmin": "min", "tmax": "max", "rhum": "mean",
    "prcp": "sum", "snow": "sum", "wdir": "mean", "wspd": "mean",
    "wpgt": "max", "pres": "mean", "tsun": "sum",
}

# Bornes physiques plausibles (contexte méditerranéen / Tunisie) :
# toute valeur hors bornes est considérée comme manquante (-> NaN -> interpolation)
BORNES_PHYSIQUES = {
    "tmax": (-25.0, 55.0),
    "tmin": (-30.0, 55.0),
    "tavg": (-30.0, 55.0),
    "pres": (940.0, 1060.0),
    "rhum": (0.0, 100.0),
    "wspd": (0.0, 200.0),
}

SOURCE_DEFAUT = "open-meteo"   # "open-meteo" | "meteostat" | "auto"


# --------------------------------------------------------------------------
# Chemins et cache
# --------------------------------------------------------------------------
def chemin_cache(source: str, lat: float, lon: float, debut: date | str, fin: date | str) -> str:
    """Construit le chemin du fichier de cache CSV (par source, zone et période)."""
    prefixe = {"open-meteo": "openmeteo", "meteostat": "meteostat"}.get(source, source)
    nom = f"{prefixe}_{lat:.3f}_{lon:.3f}_{debut}_{fin}.csv"
    return os.path.join(DATA_DIR, nom)


def _lire_cache(chemin: str) -> pd.DataFrame:
    """Charge un cache CSV et renvoie un DataFrame indexé par date."""
    df = pd.read_csv(chemin, parse_dates=["date"], index_col="date")
    df.index.name = "date"
    return df


def _chemin_existant(chemin: str) -> str | None:
    """
    Renvoie `chemin` s'il existe, sinon son équivalent à l'ancienne
    convention de nommage (`openmeteo_` ↔ `open_meteo_`), sinon None.
    Assure la compatibilité avec les caches créés par les versions
    antérieures du projet.
    """
    if os.path.exists(chemin):
        return chemin
    nom = os.path.basename(chemin)
    dossier = os.path.dirname(chemin)
    for ancien, nouveau in (("openmeteo_", "open_meteo_"), ("open_meteo_", "openmeteo_")):
        if nom.startswith(ancien):
            variante = os.path.join(dossier, nouveau + nom[len(ancien):])
            if os.path.exists(variante):
                return variante
    return None


# --------------------------------------------------------------------------
# Backend 1 : Open-Meteo
# --------------------------------------------------------------------------
def _requete_open_meteo(base_url: str, lat: float, lon: float, debut: str, fin: str) -> pd.DataFrame | None:
    """Interroge une API Open-Meteo et renvoie un DataFrame journalier mappé, ou None."""
    params = {
        "latitude": f"{lat:.4f}",
        "longitude": f"{lon:.4f}",
        "start_date": debut,
        "end_date": fin,
        "daily": ",".join(VARIABLES_OPEN_METEO),
        "timezone": "auto",   # jours civils locaux de la zone demandée
    }
    url = f"{base_url}?{urllib.parse.urlencode(params)}"
    try:
        with urllib.request.urlopen(url, timeout=60) as reponse:
            contenu = json.loads(reponse.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as exc:
        warnings.warn(f"Open-Meteo ({base_url.split('//')[1].split('/')[0]}) inaccessible : {exc}")
        return None

    quotidien = contenu.get("daily")
    if not quotidien or not quotidien.get("time"):
        return None

    df = pd.DataFrame(quotidien).rename(columns={"time": "date"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    df = df.rename(columns=VARIABLES_OPEN_METEO)
    return df[[c for c in COLONNES_SORTIE if c in df.columns]]


def _telecharger_open_meteo(lat: float, lon: float, debut: date, fin: date) -> pd.DataFrame | None:
    """
    Télécharge l'historique Open-Meteo :
      - Archive ERA5 pour [debut, fin] ;
      - si l'archive s'arrête avant `fin` (derniers jours non encore publiés),
        complète la queue avec l'API de prévision (données récentes ~92 j).
    """
    df = _requete_open_meteo(URL_ARCHIVE, lat, lon, debut.isoformat(), fin.isoformat())
    derniere = df.index.max().date() if df is not None and not df.empty else None

    if derniere is not None and derniere < fin:
        debut_queue = derniere + timedelta(days=1)
        queue = _requete_open_meteo(
            URL_PREVISION, lat, lon, debut_queue.isoformat(), fin.isoformat()
        )
        if queue is not None and not queue.empty:
            df = pd.concat([df, queue])
    elif df is None or df.empty:
        # Archive totalement indisponible (ex. dates très récentes) : prévision seule
        df = _requete_open_meteo(URL_PREVISION, lat, lon, debut.isoformat(), fin.isoformat())

    if df is None or df.empty:
        return None
    return df[~df.index.duplicated(keep="first")].sort_index()


# --------------------------------------------------------------------------
# Backend 2 : Meteostat (repli optionnel)
# --------------------------------------------------------------------------
def _telecharger_meteostat(lat: float, lon: float, debut: date, fin: date) -> pd.DataFrame | None:
    """
    Repli via la bibliothèque `meteostat` (>= 2.x) : données horaires
    agrégées en journalier. Renvoie None si la bibliothèque est absente
    ou si le réseau Meteostat est inaccessible.
    """
    try:
        from meteostat import Point, hourly  # import tardif : dépendance optionnelle
    except ImportError:
        warnings.warn("Bibliothèque `meteostat` absente — repli impossible.")
        return None

    try:
        horaire = hourly(Point(lat, lon), debut, fin).fetch()
        if horaire is None or horaire.empty:
            return None
        if "temp" in horaire.columns:
            horaire = horaire.rename(columns={"temp": "tavg"})
        agg = {k: v for k, v in AGGREGATION_HORAIRE.items() if k in horaire.columns}
        return horaire.resample("D").agg(agg)
    except Exception as exc:  # noqa: BLE001 — toute erreur réseau/parse déclenche le repli
        warnings.warn(f"Meteostat inaccessible : {exc}")
        return None


# --------------------------------------------------------------------------
# Nettoyage
# --------------------------------------------------------------------------
def nettoyer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie un DataFrame météo journalier :
      - colonnes manquantes créées (NaN) ;
      - valeurs hors bornes physiques converties en NaN ;
      - interpolation linéaire des séries numériques (limitée à 7 jours) ;
      - précipitations manquantes comblées par 0 (pas d'observation = pas de pluie) ;
      - tri chronologique, suppression des doublons et des lignes sans température.
    """
    df = df.copy()
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]

    for col in COLONNES_SORTIE:
        if col not in df.columns:
            df[col] = np.nan

    # Les valeurs aberrantes deviennent NaN avant interpolation
    for col, (borne_min, borne_max) in BORNES_PHYSIQUES.items():
        df[col] = df[col].clip(borne_min, borne_max)

    # Interpolation linéaire des séries continues
    colonnes_interpolables = ["tavg", "tmin", "tmax", "pres", "rhum", "wspd", "wdir"]
    df[colonnes_interpolables] = df[colonnes_interpolables].interpolate(method="time", limit=7)

    # Précipitations / neige : NaN -> 0 (absence d'observation = pas de pluie)
    for col in ("prcp", "snow"):
        if col in df.columns:
            df[col] = df[col].fillna(0.0)

    # Supprime les journées sans aucune température exploitable
    df = df.dropna(subset=["tmax", "tmin"], how="all")
    return df[COLONNES_SORTIE]


# --------------------------------------------------------------------------
# API publique
# --------------------------------------------------------------------------
def fetch_data(
    lat: float,
    lon: float,
    debut: date | str,
    fin: date | str,
    use_cache: bool = True,
    source: str = SOURCE_DEFAUT,
) -> pd.DataFrame:
    """
    Récupère l'historique météo journalier pour une zone géographique.

    Paramètres
    ----------
    lat, lon : coordonnées GPS de la zone
    debut, fin : période au format 'AAAA-MM-JJ'
    use_cache : utilise/écrit le cache CSV local (recommandé)
    source : 'open-meteo' (défaut), 'meteostat' ou 'auto' (essaie les deux)

    Retour
    ------
    DataFrame indexé par date avec les colonnes COLONNES_SORTIE.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    d_debut = pd.to_datetime(debut).date()
    d_fin = pd.to_datetime(fin).date()
    source_retenue = source if source != "auto" else SOURCE_DEFAUT

    chemin = chemin_cache(source_retenue, lat, lon, debut, fin)
    if use_cache:
        existant = _chemin_existant(chemin)
        if existant is not None:
            return _lire_cache(existant)

    # Sélection du backend
    if source == "meteostat":
        backends = [_telecharger_meteostat]
    elif source == "auto":
        backends = [_telecharger_open_meteo, _telecharger_meteostat]
    else:
        backends = [_telecharger_open_meteo]

    df = None
    for backend in backends:
        df = backend(lat, lon, d_debut, d_fin)
        if df is not None and not df.empty:
            source_retenue = backend.__name__.replace("_telecharger_", "")
            break

    if df is None or df.empty:
        raise RuntimeError(
            f"Aucune donnée météo récupérable pour ({lat:.3f}, {lon:.3f}) "
            f"entre {debut} et {fin}. Vérifiez votre connexion réseau."
        )

    df = nettoyer(df)
    if df.empty:
        raise RuntimeError("Les données téléchargées sont vides après nettoyage.")

    chemin = chemin_cache(source_retenue, lat, lon, debut, fin)
    df.to_csv(chemin, index_label="date")
    return df


def charger_donnees_ville(
    ville: str,
    debut: date | str = "2010-01-01",
    fin: date | str | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Raccourci : historique complet d'une ville prédéfinie (voir VILLES)."""
    ville_norm = normaliser_ville(ville)
    if ville_norm not in VILLES:
        raise KeyError(f"Ville inconnue : {ville}. Disponibles : {list(VILLES)}")
    lat, lon = VILLES[ville_norm]
    fin = fin or (date.today() - timedelta(days=1)).isoformat()
    return fetch_data(lat, lon, debut, fin, use_cache=use_cache)


if __name__ == "__main__":
    # Petit test manuel : python -m src.data_loader
    print("Téléchargement de l'historique de Sousse (1ère exécution : ~20-30 s)...")
    data = charger_donnees_ville("Sousse", debut="2015-01-01")
    print(data.tail())
    print(f"\n{len(data)} jours | période : {data.index.min().date()} -> {data.index.max().date()}")
