"""
predict.py
==========
Inférence : prévisions multi-jours (1 à 7 jours) de la plateforme.

Pour chaque horizon h = 1..N, le système prédit :
  - la température maximale   (`tmax`) + son intervalle de confiance 90 % ;
  - la température minimale   (`tmin`) ;
  - l'humidité relative       (`rhum`) ;
  - la vitesse du vent        (`wspd`) ;
  - la pression               (`pres`) ;
  - la probabilité de pluie   (modèle de classification).

Principe de la prévision multi-jours (ROULAGE RÉCURSIF)
-------------------------------------------------------
1. Récupération de l'historique RÉEL récent (60 jours, jusqu'à aujourd'hui
   via l'API de prévision Open-Meteo) ;
2. Pour l'horizon 1, les features (lags, moyennes mobiles) sont alimentées
   par les observations réelles — condition de déploiement idéale ;
3. Pour les horizons suivants, les valeurs brutes sont INCONNUES : elles
   sont remplacées par les prédictions du tour précédent, réinjectées dans
   le calcul des features (schéma autorégressif, proche de la façon dont
   opèrent réellement les modèles météorologiques ensemblistes).

L'intervalle de confiance s'élargit avec l'horizon (approximation
marche aléatoire : largeur × √h), ce qui reflète l'incertitude croissante.

Usage :
  python -m src.predict --ville Sousse --jours 5
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from datetime import date, timedelta

import joblib
import numpy as np
import pandas as pd

from src import climatology
from src.data_loader import VILLES, fetch_data, normaliser_ville
from src.feature_engineering import construire_features
from src.train import (
    BORNES_CIBLES,
    CIBLES_REGRESSION,
    CIBLES_REQUIS,
    SEUIL_PLUIE_MM,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELES_DIR = os.path.join(BASE_DIR, "models")

HISTORIQUE_RECENT_JOURS = 60
HORIZON_MAX = 7


# --------------------------------------------------------------------------
# Chargement des artefacts
# --------------------------------------------------------------------------
def charger_modele(tache: str, ville: str) -> dict:
    """Charge un bundle modèle sauvegardé par train.py ('tmax', 'tmin', ..., 'rain')."""
    chemin = os.path.join(MODELES_DIR, f"{tache}_{ville}.pkl")
    if not os.path.exists(chemin):
        raise FileNotFoundError(
            f"Aucun modèle '{tache}' pour {ville} ({chemin}). "
            f"Exécutez d'abord : python -m src.train --ville {ville}"
        )
    return joblib.load(chemin)


def charger_infos(ville: str) -> dict | None:
    """Charge les métadonnées d'entraînement (info_<ville>.json), ou None."""
    ville = normaliser_ville(ville)
    chemin = os.path.join(MODELES_DIR, f"info_{ville}.json")
    if not os.path.exists(chemin):
        return None
    with open(chemin, encoding="utf-8") as f:
        return json.load(f)


def modele_existe(tache: str, ville: str) -> bool:
    """Vérifie qu'un modèle sauvegardé est disponible."""
    ville = normaliser_ville(ville)
    return os.path.exists(os.path.join(MODELES_DIR, f"{tache}_{ville}.pkl"))


def modeles_complets(ville: str) -> bool:
    """Vrai si TOUS les modèles requis (multi-variables + pluie) sont présents."""
    return all(modele_existe(c, ville) for c in CIBLES_REQUIS)


# --------------------------------------------------------------------------
# Climatologie
# --------------------------------------------------------------------------
def moyenne_historique_tmax(df: pd.DataFrame, date_cible: pd.Timestamp, fenetre: int = 7) -> float:
    """
    Climatologie : moyenne de `tmax` pour le même jour de l'année
    sur toute l'historique, +/- `fenetre` jours (distance circulaire).
    """
    doy_cible = date_cible.dayofyear
    doy = df.index.dayofyear.to_numpy()
    distance = np.minimum(np.abs(doy - doy_cible), 365 - np.abs(doy - doy_cible))
    voisinage = df.loc[distance <= fenetre, "tmax"]
    return float(voisinage.mean()) if len(voisinage) else float("nan")


def _charger_climatologie(lat: float, lon: float, infos: dict | None) -> pd.DataFrame:
    """Historique long (normales de saison) réutilisant les caches locaux."""
    debut = "2010-01-01"
    if infos and infos.get("periode", {}).get("debut"):
        debut = f"{str(infos['periode']['debut'])[:4]}-01-01"
    try:
        return climatology.charger_historique(lat, lon, debut=debut)
    except Exception as exc:  # noqa: BLE001 — climatologie optionnelle
        warnings.warn(f"Historique climatologique indisponible : {exc}")
        return pd.DataFrame()


# --------------------------------------------------------------------------
# Prévision multi-jours
# --------------------------------------------------------------------------
def _resoudre_localisation(
    ville: str | None, lat: float | None, lon: float | None
) -> tuple[str, float, float]:
    """Normalise (ville, lat, lon) à partir de l'un ou l'autre des modes."""
    if ville is None and (lat is None or lon is None):
        raise ValueError("Fournissez `ville` ou bien `lat` et `lon`.")
    if ville is None:
        ville = f"GPS_{lat:.2f}_{lon:.2f}"
    else:
        ville = normaliser_ville(ville)
    if lat is None or lon is None:
        if ville not in VILLES:
            raise KeyError(f"Ville inconnue : {ville}. Disponibles : {list(VILLES)}")
        lat, lon = VILLES[ville]
    return ville, float(lat), float(lon)


def predire_multijours(
    ville: str | None = None,
    lat: float | None = None,
    lon: float | None = None,
    jours: int = HORIZON_MAX,
) -> dict:
    """
    Prédit tmax (± IC 90 %), tmin, humidité, vent, pression et risque de
    pluie pour les `jours` prochains jours (1 à 7).

    Paramètres : soit `ville` (VILLES prédéfinies), soit `lat` + `lon`.

    Retour : dictionnaire contenant la liste `previsions` (une entrée par
    jour, avec le vecteur de features utilisé), les métriques des modèles,
    les intervalles et, pour compatibilité, les clés du lendemain.
    """
    ville, lat, lon = _resoudre_localisation(ville, lat, lon)
    jours = int(max(1, min(jours, HORIZON_MAX)))

    bundles = {cible: charger_modele(cible, ville) for cible in CIBLES_REQUIS}

    fin = date.today()
    debut = fin - timedelta(days=HISTORIQUE_RECENT_JOURS)
    df_raw = fetch_data(lat, lon, debut.isoformat(), fin.isoformat(), use_cache=False)

    derniere_obs = df_raw.index[-1]
    infos = charger_infos(ville)
    df_clim = _charger_climatologie(lat, lon, infos)
    if df_clim.empty:
        df_clim = df_raw

    ic_tmax = bundles["tmax"].get("intervalle_confiance")
    seuil_pluie = bundles["rain"].get("seuil_pluie_mm", SEUIL_PLUIE_MM)

    previsions: list[dict] = []
    for horizon in range(1, jours + 1):
        date_cible = derniere_obs + pd.Timedelta(days=horizon)
        df_raw.loc[date_cible] = np.nan
        df_features = construire_features(df_raw)
        entree = df_features.loc[[date_cible]]

        valeurs: dict[str, float] = {}
        for cible in CIBLES_REGRESSION:
            bundle = bundles[cible]
            X = entree[bundle["features"]].fillna(0)
            prediction = float(bundle["model"].predict(X)[0])
            borne = BORNES_CIBLES.get(cible)
            if borne is not None:
                prediction = float(np.clip(prediction, borne[0], borne[1]))
            valeurs[cible] = prediction

        bundle_rain = bundles["rain"]
        X_rain = entree[bundle_rain["features"]].fillna(0)
        proba_pluie = float(bundle_rain["model"].predict_proba(X_rain)[0, 1])

        # Réinjection des prédictions dans l'historique pour l'horizon suivant
        for cible, valeur in valeurs.items():
            df_raw.loc[date_cible, cible] = valeur
        df_raw.loc[date_cible, "prcp"] = seuil_pluie if proba_pluie >= 0.5 else 0.0

        # Intervalle de confiance élargi avec l'horizon
        if ic_tmax:
            largeur = np.sqrt(horizon)
            ic_bas = valeurs["tmax"] + float(ic_tmax["q_bas"]) * largeur
            ic_haut = valeurs["tmax"] + float(ic_tmax["q_haut"]) * largeur
        else:
            ic_bas = ic_haut = None

        tmax_moyenne = moyenne_historique_tmax(df_clim.dropna(subset=["tmax"]), date_cible)
        ecart = valeurs["tmax"] - tmax_moyenne if not np.isnan(tmax_moyenne) else float("nan")

        previsions.append({
            "date": str(date_cible.date()),
            "horizon": horizon,
            "tmax": round(valeurs["tmax"], 1),
            "ic_bas": round(ic_bas, 1) if ic_bas is not None else None,
            "ic_haut": round(ic_haut, 1) if ic_haut is not None else None,
            "tmin": round(valeurs["tmin"], 1),
            "rhum": round(valeurs["rhum"], 0),
            "wspd": round(valeurs["wspd"], 0),
            "pres": round(valeurs["pres"], 0),
            "proba_pluie": round(proba_pluie, 3),
            "risque_pluie": bool(proba_pluie >= 0.5),
            "tmax_moyenne_historique": round(tmax_moyenne, 1) if not np.isnan(tmax_moyenne) else None,
            "ecart_moyenne": round(ecart, 1) if not np.isnan(ecart) else None,
            "features_row": entree,
        })

    tmax_dernier = df_raw.loc[derniere_obs, "tmax"]
    premier = previsions[0]

    return {
        "ville": ville,
        "lat": lat,
        "lon": lon,
        "jours": jours,
        "date_derniere_observation": str(derniere_obs.date()),
        "date_prediction": premier["date"],
        "tmax_dernier_obs": float(tmax_dernier) if not pd.isna(tmax_dernier) else None,
        "previsions": previsions,
        "seuil_pluie_mm": seuil_pluie,
        "modeles": {c: bundles[c]["nom_modele"] for c in CIBLES_REQUIS},
        "metriques": {c: bundles[c]["metriques_test"] for c in CIBLES_REQUIS},
        "intervalles": {c: bundles[c].get("intervalle_confiance") for c in CIBLES_REGRESSION},
        "infos": infos,
        # Compatibilité (horizon 1)
        "tmax_prevue": premier["tmax"],
        "tmax_moyenne_historique": premier["tmax_moyenne_historique"],
        "ecart_moyenne": premier["ecart_moyenne"],
        "proba_pluie": premier["proba_pluie"],
        "risque_pluie": premier["risque_pluie"],
        "modele_tmax": bundles["tmax"]["nom_modele"],
        "modele_rain": bundles["rain"]["nom_modele"],
        "metriques_tmax": bundles["tmax"]["metriques_test"],
        "metriques_rain": bundles["rain"]["metriques_test"],
    }


def predire_lendemain(
    ville: str | None = None,
    lat: float | None = None,
    lon: float | None = None,
) -> dict:
    """Prévision du lendemain (horizon 1) — raccourci de `predire_multijours`."""
    return predire_multijours(ville=ville, lat=lat, lon=lon, jours=1)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prévision météo (1 à 7 jours)")
    parser.add_argument("--ville", default="Sousse")
    parser.add_argument("--lat", type=float, default=None)
    parser.add_argument("--lon", type=float, default=None)
    parser.add_argument("--jours", type=int, default=1, help="Horizon de prévision (1 à 7)")
    args = parser.parse_args()

    resultat = predire_multijours(
        ville=args.ville, lat=args.lat, lon=args.lon, jours=args.jours
    )

    print(f"=== Prévision météo — {resultat['ville']} ===")
    print(f"Dernier relevé : {resultat['date_derniere_observation']}")
    print(f"Modèles : {resultat['modeles']}\n")
    for p in resultat["previsions"]:
        ic = ""
        if p["ic_bas"] is not None:
            ic = f" (IC 90 % : {p['ic_bas']} -> {p['ic_haut']} °C)"
        pluie = "PLUIE" if p["risque_pluie"] else "sec"
        print(
            f"J+{p['horizon']} {p['date']} | tmax {p['tmax']:5.1f} °C{ic} | "
            f"tmin {p['tmin']:5.1f} °C | vent {p['wspd']:3.0f} km/h | "
            f"hum {p['rhum']:3.0f} % | pres {p['pres']:.0f} hPa | "
            f"pluie {p['proba_pluie'] * 100:3.0f} % ({pluie})"
        )
