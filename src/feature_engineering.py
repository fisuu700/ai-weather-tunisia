"""
feature_engineering.py
======================
Construction des variables prédictives (features) à partir de
l'historique météo journalier.

Trois familles de features sont créées :

  1. Décomposition temporelle
     - jour du mois (`day`), mois (`month`), jour de l'année (`dayofyear`)
     - saison (`season`) : 0 = hiver, 1 = printemps, 2 = été, 3 = automne
     - encodage cyclique sin/cos du jour, du mois et du jour de l'année
       (indispensable pour que le modèle comprenne la circularité du calendrier :
        le 31/12 est tout proche du 01/01)

  2. Lag features — valeurs observées les jours précédents (t-1, t-2, t-3, t-7)
     - tmax, tmin (températures), pres (pression), rhum (humidité), wspd (vent)

  3. Rolling features — moyennes mobiles sur 3 et 7 jours de `tmax` et `prcp`

IMPORTANT — Data leakage (fuite d'information)
-----------------------------------------------
Les features d'une journée J ne doivent contenir AUCUNE observation du jour J
sinon la tâche devient triviale et les performances mesurées sont irréalistes.
C'est pourquoi :
  - les lag features commencent à t-1 (jamais t) ;
  - les moyennes mobiles sont calculées après un décalage d'un jour
    (`shift(1)` avant `rolling`) : la moyenne mobile de tmax sur 3 jours
    en J correspond à la moyenne de J-3, J-2 et J-1.

Ainsi, prédire `tmax(J)` avec ces features équivaut exactement à la tâche
réelle : « prédire la température de demain à partir de tout l'historique
jusqu'à aujourd'hui ».
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Variables brutes utilisées pour les lags (conformément au cahier des charges)
COLONNES_LAG = ("tmax", "tmin", "pres", "rhum", "wspd")
LAGS = (1, 2, 3, 7)

# Variables pour les moyennes mobiles
COLONNES_ROLLING = ("tmax", "prcp")
FENETRES_ROLLING = (3, 7)


# --------------------------------------------------------------------------
# 1. Décomposition temporelle
# --------------------------------------------------------------------------
def _saisons(mois: np.ndarray) -> np.ndarray:
    """Mappe le mois (1-12) vers la saison : 0 hiver, 1 printemps, 2 été, 3 automne."""
    return np.select(
        [mois.isin([12, 1, 2]), mois.isin([3, 4, 5]), mois.isin([6, 7, 8])],
        [0, 1, 2],
        default=3,
    )


def ajouter_features_temporelles(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute les variables de calendrier et leurs encodages cycliques sin/cos."""
    df = df.copy()
    index = df.index

    df["day"] = index.day
    df["month"] = index.month
    df["dayofyear"] = index.dayofyear
    df["year"] = index.year
    df["season"] = _saisons(pd.Series(index.month))

    # Encodage cyclique : sin/cos du jour de l'année, du mois et du jour
    df["dayofyear_sin"] = np.sin(2 * np.pi * df["dayofyear"] / 365.25)
    df["dayofyear_cos"] = np.cos(2 * np.pi * df["dayofyear"] / 365.25)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)
    df["day_sin"] = np.sin(2 * np.pi * df["day"] / 31.0)
    df["day_cos"] = np.cos(2 * np.pi * df["day"] / 31.0)
    return df


# --------------------------------------------------------------------------
# 2. Lag features
# --------------------------------------------------------------------------
def ajouter_features_lag(
    df: pd.DataFrame,
    colonnes: tuple[str, ...] = COLONNES_LAG,
    lags: tuple[int, ...] = LAGS,
) -> pd.DataFrame:
    """
    Ajoute pour chaque colonne et chaque décalage t-k la colonne `<col>_lag_<k>`
    contenant la valeur observée k jours plus tôt.

    Exemple : `tmax_lag_1` en date J contient la valeur de `tmax` au jour J-1.
    Une colonne entièrement manquante (ex. `rhum` si indisponible) est ignorée.
    """
    df = df.copy()
    for col in colonnes:
        if col not in df.columns or df[col].isna().all():
            continue  # colonne absente ou vide : on ne crée pas ses lags
        for lag in lags:
            df[f"{col}_lag_{lag}"] = df[col].shift(lag)
    return df


# --------------------------------------------------------------------------
# 3. Rolling features
# --------------------------------------------------------------------------
def ajouter_features_rolling(
    df: pd.DataFrame,
    colonnes: tuple[str, ...] = COLONNES_ROLLING,
    fenetres: tuple[int, ...] = FENETRES_ROLLING,
) -> pd.DataFrame:
    """
    Ajoute les moyennes mobiles `<col>_roll_<n>` = moyenne de `col` sur les
    n jours PRÉCÉDENTS (le jour courant est exclu via `shift(1)`, cf. leakage).

    Exemple : `tmax_roll_7` en date J = moyenne de tmax sur J-7 ... J-1.
    """
    df = df.copy()
    for col in colonnes:
        if col not in df.columns or df[col].isna().all():
            continue
        decale = df[col].shift(1)  # exclut le jour courant
        for fenetre in fenetres:
            df[f"{col}_roll_{fenetre}"] = decale.rolling(window=fenetre, min_periods=fenetre).mean()
    return df


# --------------------------------------------------------------------------
# Pipeline complet
# --------------------------------------------------------------------------
def construire_features(df: pd.DataFrame) -> pd.DataFrame:
    """Applique les trois étapes de feature engineering dans l'ordre."""
    df = ajouter_features_temporelles(df)
    df = ajouter_features_lag(df)
    df = ajouter_features_rolling(df)
    return df


# Liste ordonnée des features candidates (hors colonnes brutes du DataFrame)
FEATURES_CANDIDATES: list[str] = [
    # Calendrier
    "day", "month", "dayofyear", "season",
    "dayofyear_sin", "dayofyear_cos", "month_sin", "month_cos", "day_sin", "day_cos",
    # Lags températures
    "tmax_lag_1", "tmax_lag_2", "tmax_lag_3", "tmax_lag_7",
    "tmin_lag_1", "tmin_lag_2", "tmin_lag_3", "tmin_lag_7",
    # Lags pression
    "pres_lag_1", "pres_lag_2", "pres_lag_3", "pres_lag_7",
    # Lags humidité
    "rhum_lag_1", "rhum_lag_2", "rhum_lag_3", "rhum_lag_7",
    # Lags vent
    "wspd_lag_1", "wspd_lag_2", "wspd_lag_3", "wspd_lag_7",
    # Moyennes mobiles
    "tmax_roll_3", "tmax_roll_7", "prcp_roll_3", "prcp_roll_7",
]


def features_disponibles(df: pd.DataFrame) -> list[str]:
    """
    Filtre la liste des features candidates pour ne garder que celles
    réellement présentes et renseignées dans `df`.

    Permet au pipeline de s'adapter automatiquement si une variable brute
    est indisponible (ex. `rhum` absent du repli journalier Meteostat).
    """
    return [
        f for f in FEATURES_CANDIDATES
        if f in df.columns and not df[f].isna().all()
    ]
