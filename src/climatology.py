"""
climatology.py
==============
Analyse climatologique avancée de l'historique météo d'un point :

  - `charger_historique`     — historique long (années) avec réutilisation
    intelligente des caches CSV locaux ;
  - `normales_mensuelles`    — normales de saison par mois (tmax, tmin,
    pluie cumulée, humidité, vent, pression) ;
  - `normales_saisonnières`  — idem agrégées par saison météorologique ;
  - `anomalies_annuelles`    — écart de la moyenne annuelle de tmax
    par rapport à la moyenne de référence (réchauffement local).

Ces agrégats alimentent l'onglet « Climatologie » du dashboard et
permettent de situer une prévision par rapport aux normales de saison.
"""

from __future__ import annotations

import glob
import os
import warnings
from datetime import date, timedelta

import numpy as np
import pandas as pd

from src.data_loader import DATA_DIR, fetch_data

MOIS_FR = {
    1: "Janv", 2: "Févr", 3: "Mars", 4: "Avr", 5: "Mai", 6: "Juin",
    7: "Juil", 8: "Août", 9: "Sept", 10: "Oct", 11: "Nov", 12: "Déc",
}

SAISONS_FR = {0: "Hiver", 1: "Printemps", 2: "Été", 3: "Automne"}


def _saisons_mois(mois: pd.Series) -> pd.Series:
    """Mappe le mois (1-12) vers la saison météorologique (0 hiver → 3 automne)."""
    return pd.Series(
        np.select(
            [mois.isin([12, 1, 2]), mois.isin([3, 4, 5]), mois.isin([6, 7, 8])],
            [0, 1, 2],
            default=3,
        ),
        index=mois.index,
    )


def _lire_cache_le_plus_recent(lat: float, lon: float, debut: str) -> pd.DataFrame | None:
    """
    Recherche un cache CSV Open-Meteo quelconque pour (lat, lon, debut).
    Permet de réutiliser un historique téléchargé la veille (les normales
    mensuelles ne changent pas de façon notable d'un jour à l'autre).
    """
    motif = os.path.join(DATA_DIR, f"*meteo_{lat:.3f}_{lon:.3f}_{debut}_*.csv")
    fichiers = sorted(glob.glob(motif), key=os.path.getmtime, reverse=True)
    if not fichiers:
        return None
    try:
        df = pd.read_csv(fichiers[0], parse_dates=["date"], index_col="date")
        df.index.name = "date"
        return df
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        warnings.warn(f"Cache climatologique illisible ({fichiers[0]}) : {exc}")
        return None


def charger_historique(
    lat: float,
    lon: float,
    debut: str = "2010-01-01",
    fin: str | None = None,
) -> pd.DataFrame:
    """
    Historique long et nettoyé d'un point (usage climatologique).

    Stratégie de cache :
      1. cache exact (même période) → lecture immédiate ;
      2. cache d'une période antérieure débutant à la même date →
         réutilisé tel quel (suffisant pour les normales) ;
      3. sinon téléchargement complet via Open-Meteo.
    """
    fin = fin or (date.today() - timedelta(days=1)).isoformat()

    exact = os.path.join(DATA_DIR, f"openmeteo_{lat:.3f}_{lon:.3f}_{debut}_{fin}.csv")
    legacy = os.path.join(DATA_DIR, f"open_meteo_{lat:.3f}_{lon:.3f}_{debut}_{fin}.csv")
    if os.path.exists(exact) or os.path.exists(legacy):
        return fetch_data(lat, lon, debut, fin, use_cache=True)

    ancien = _lire_cache_le_plus_recent(lat, lon, debut)
    if ancien is not None and len(ancien) > 365:
        return ancien

    return fetch_data(lat, lon, debut, fin, use_cache=True)


def normales_mensuelles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normales mensuelles sur toute la période du DataFrame.

    Retour (index = mois 1..12) :
      tmax, tmin  : moyenne des moyennes mensuelles ;
      prcp        : moyenne des TOTAUX mensuels de pluie (mm) ;
      rhum, wspd, pres : moyennes.
    """
    travail = df.copy()
    travail["month"] = travail.index.month
    travail["year"] = travail.index.year

    pluie_mensuelle = travail.groupby(["year", "month"])["prcp"].sum()

    return pd.DataFrame({
        "tmax": travail.groupby("month")["tmax"].mean(),
        "tmin": travail.groupby("month")["tmin"].mean(),
        "prcp": pluie_mensuelle.groupby("month").mean(),
        "rhum": travail.groupby("month")["rhum"].mean(),
        "wspd": travail.groupby("month")["wspd"].mean(),
        "pres": travail.groupby("month")["pres"].mean(),
    })


def normales_saisonnières(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normales par saison météorologique (index 0..3 : hiver → automne).

    Mêmes agrégats que `normales_mensuelles`, la pluie restant un
    cumul moyen par saison (en mm).
    """
    travail = df.copy()
    travail["season"] = _saisons_mois(pd.Series(travail.index.month, index=travail.index))
    travail["year"] = travail.index.year

    pluie_saison = travail.groupby(["year", "season"])["prcp"].sum()

    table = pd.DataFrame({
        "tmax": travail.groupby("season")["tmax"].mean(),
        "tmin": travail.groupby("season")["tmin"].mean(),
        "prcp": pluie_saison.groupby("season").mean(),
        "rhum": travail.groupby("season")["rhum"].mean(),
        "wspd": travail.groupby("season")["wspd"].mean(),
    })
    table.index = [SAISONS_FR.get(i, str(i)) for i in table.index]
    return table


def anomalies_annuelles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Écart de la moyenne annuelle de `tmax` par rapport à la moyenne
    générale de la période (référence climatologique locale).

    Retour : DataFrame indexé par année, colonnes tmax_moy et ecart (°C).
    """
    travail = df.dropna(subset=["tmax"]).copy()
    par_annee = travail.groupby(travail.index.year)["tmax"].mean()
    reference = float(par_annee.mean())
    return pd.DataFrame({
        "tmax_moy": par_annee,
        "ecart": par_annee - reference,
        "reference": reference,
    })


def mois_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Copie du DataFrame enrichie des colonnes `month` et `month_label` (pour boîtes à moustaches)."""
    travail = df.copy()
    travail["month"] = travail.index.month
    travail["month_label"] = [MOIS_FR[m] for m in travail["month"]]
    return travail
