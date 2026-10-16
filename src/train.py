"""
train.py
========
Entraînement, comparaison et sauvegarde des modèles de prédiction météo.

Cibles modélisées pour chaque ville :
  · RÉGRESSION (RandomForest / GradientBoosting / XGBoost)
      - température maximale demain  (`tmax`)
      - température minimale demain  (`tmin`)
      - humidité relative demain     (`rhum`)
      - vitesse du vent demain       (`wspd`)
      - pression au niveau de la mer (`pres`)
  · CLASSIFICATION — risque de pluie demain (`prcp` > 1 mm)
      - RandomForest / GradientBoosting / XGBoost

Chaque modèle de régression est accompagné d'un INTERVALLE DE CONFIANCE
à 90 % estimé par les quantiles résiduels sur le jeu de test (méthode
« percentile », sans hypothèse de normalité des erreurs).

Découpage TEMPOREL des données (aucun mélange aléatoire : interdit pour
une série temporelle !) : les dernières `annees_test` années servent de
jeu de test, ce qui reproduit la situation réelle « entraîner sur le
passé, prédire le futur ».

Artefacts produits :
  models/<cible>_<ville>.pkl    — meilleur modèle par cible (joblib)
  models/info_<ville>.json      — métriques, features, période, intervalles
  data/test_preds_<ville>.csv   — réel vs prédit sur le jeu de test

Usage en ligne de commande :
  python -m src.train --ville Sousse --debut 2010-01-01 --annees-test 2
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import date, timedelta

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier, XGBRegressor

from src.data_loader import VILLES, fetch_data, normaliser_ville
from src.feature_engineering import construire_features, features_disponibles

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELES_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")

RANDOM_STATE = 42
SEUIL_PLUIE_MM = 1.0   # un jour est considéré « pluvieux » si prcp > 1 mm
ANNEES_TEST_DEFAUT = 2

# Cibles de régression (dans l'ordre d'entraînement)
CIBLES_REGRESSION: tuple[str, ...] = ("tmax", "tmin", "rhum", "wspd", "pres")
# Toutes les cibles nécessaires au fonctionnement complet de la plateforme
CIBLES_REQUIS: tuple[str, ...] = (*CIBLES_REGRESSION, "rain")

# Bornes physiques appliquées aux prédictions (cohérence météo)
BORNES_CIBLES: dict[str, tuple[float, float]] = {
    "tmax": (-25.0, 55.0),
    "tmin": (-30.0, 55.0),
    "rhum": (0.0, 100.0),
    "wspd": (0.0, 200.0),
    "pres": (940.0, 1060.0),
}

# Niveau de l'intervalle de confiance (quantiles résiduels symétriques par construction)
NIVEAU_INTERVALLE = 0.90


# --------------------------------------------------------------------------
# Préparation des données
# --------------------------------------------------------------------------
def preparer_donnees(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, pd.Series], pd.Series, list[str]]:
    """
    Construit la matrice de features et les cibles de régression + pluie.

    Retour
    ------
    X       : matrice de features (index temporel conservé) ;
    cibles  : dict {variable: série cible} pour tmax, tmin, rhum, wspd, pres ;
    y_rain  : 1 si jour pluvieux (prcp > 1 mm), sinon 0 ;
    features: liste des features effectivement utilisées.
    """
    df = construire_features(df)
    features = features_disponibles(df)

    colonnes_cibles = [
        c for c in CIBLES_REGRESSION
        if c in df.columns and not df[c].isna().all()
    ]
    donnees = df[features + colonnes_cibles].dropna()
    X = donnees[features]

    cibles = {c: donnees[c] for c in colonnes_cibles}
    y_rain = (df.loc[donnees.index, "prcp"] > SEUIL_PLUIE_MM).astype(int)
    return X, cibles, y_rain, features


def decoupe_temporelle(
    X: pd.DataFrame,
    y: pd.Series,
    annees_test: int = ANNEES_TEST_DEFAUT,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Découpage temporel : les dernières `annees_test` années => jeu de test.
    Aucun shuffle — l'ordre chronologique est préservé.
    """
    limite = X.index.max() - pd.DateOffset(years=annees_test)
    train_mask = X.index <= limite
    test_mask = ~train_mask
    return X[train_mask], X[test_mask], y[train_mask], y[test_mask]


# --------------------------------------------------------------------------
# Métriques
# --------------------------------------------------------------------------
def evaluer_regression(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    """MAE, RMSE et R² pour la régression de la température maximale."""
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def intervalle_confiance(
    y_true: pd.Series,
    y_pred: np.ndarray,
    niveau: float = NIVEAU_INTERVALLE,
) -> dict[str, float | str]:
    """
    Intervalle de confiance par quantiles résiduels sur le jeu de test.

    Pour un niveau `niveau` (ex. 0,90), l'intervalle autour d'une
    prédiction est [prédiction + q_bas, prédiction + q_haut], où q_bas et
    q_haut sont les quantiles (1-niveau)/2 et (1+niveau)/2 des résidus
    observés (y_true - y_pred). Méthode « percentile », robuste et sans
    hypothèse de normalité des erreurs.
    """
    residus = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    alpha = (1.0 - niveau) / 2.0
    return {
        "niveau": float(niveau),
        "q_bas": float(np.quantile(residus, alpha)),
        "q_haut": float(np.quantile(residus, 1.0 - alpha)),
        "std": float(np.std(residus)),
        "mae": float(np.mean(np.abs(residus))),
        "methode": "quantiles résiduels (jeu de test)",
    }


def evaluer_classification(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
) -> dict[str, float]:
    """Accuracy, Precision, Recall, F1 (et AUC si probabilités fournies)."""
    metriques = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_proba is not None and len(np.unique(y_true)) > 1:
        metriques["auc"] = float(roc_auc_score(y_true, y_proba))
    return metriques


# --------------------------------------------------------------------------
# Catalogues de modèles
# --------------------------------------------------------------------------
def modeles_regression() -> dict:
    """Trois régresseurs comparés pour la prédiction de tmax."""
    return {
        "RandomForest": RandomForestRegressor(
            n_estimators=400, min_samples_leaf=2,
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=3,
            random_state=RANDOM_STATE,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=400, learning_rate=0.05, max_depth=4,
            subsample=0.9, colsample_bytree=0.9,
            random_state=RANDOM_STATE, n_jobs=-1, verbosity=0,
        ),
    }


def modeles_classification() -> dict:
    """Classifieurs comparés pour le risque de pluie."""
    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=400, min_samples_leaf=3, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=3,
            random_state=RANDOM_STATE,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=4,
            subsample=0.9, colsample_bytree=0.9, scale_pos_weight=3,
            random_state=RANDOM_STATE, n_jobs=-1, verbosity=0,
        ),
    }


# --------------------------------------------------------------------------
# Boucle d'entraînement
# --------------------------------------------------------------------------
def _entrainer_et_comparer(
    modeles: dict,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    tache: str,
    verbose: bool = True,
) -> tuple[object, str, dict[str, float], pd.DataFrame]:
    """
    Entraîne chaque modèle, calcule ses métriques et identifie le meilleur.

    Retour : (meilleur_modèle, nom, métriques_test, tableau comparatif).
    """
    resultats, modeles_entraines = [], {}
    predict_proba_dispo = tache == "classification"

    for nom, modele in modeles.items():
        modele.fit(X_train, y_train)
        pred_train = modele.predict(X_train)
        pred_test = modele.predict(X_test)

        if tache == "regression":
            met_test = evaluer_regression(y_test, pred_test)
            met_train = evaluer_regression(y_train, pred_train)
        else:
            proba = modele.predict_proba(X_test)[:, 1] if predict_proba_dispo else None
            met_test = evaluer_classification(y_test, pred_test, proba)
            met_train = evaluer_classification(y_train, pred_train)

        modeles_entraines[nom] = modele
        resultats.append({"modèle": nom, **met_test})
        if verbose:
            nom_metrique = "MAE" if tache == "regression" else "f1"
            print(f"  [{nom:<16}] test {nom_metrique} = {met_test[nom_metrique]:.4f} | "
                  f"train {nom_metrique} = {met_train[nom_metrique]:.4f}")

    tableau = pd.DataFrame(resultats).set_index("modèle")

    # Sélection du meilleur modèle : MAE minimale (régression) / F1 maximale (classification)
    critere = "MAE" if tache == "regression" else "f1"
    meilleur_nom = tableau[critere].idxmin() if tache == "regression" else tableau[critere].idxmax()

    if verbose:
        print(f"\n  Tableau comparatif ({tache}) :")
        print(tableau.round(4).to_string())
        print(f"  >>> Meilleur modèle : {meilleur_nom}\n")

    return modeles_entraines[meilleur_nom], meilleur_nom, tableau.loc[meilleur_nom].to_dict(), tableau


# --------------------------------------------------------------------------
# Entraînement complet pour une ville
# --------------------------------------------------------------------------
def entrainer_et_sauvegarder(
    df: pd.DataFrame,
    ville: str,
    lat: float,
    lon: float,
    annees_test: int = ANNEES_TEST_DEFAUT,
    verbose: bool = True,
) -> dict:
    """
    Pipeline complet à partir d'un DataFrame météo déjà nettoyé :
    features -> split temporel -> comparaison des modèles ->
    sauvegarde des meilleurs modèles (joblib) + métriques (JSON) + prédictions test.
    """
    os.makedirs(MODELES_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    X, cibles, y_rain, features = preparer_donnees(df)
    if verbose:
        print(f"Dataset {ville} : {len(X)} jours | features : {len(features)}")
        print(f"Période : {X.index.min().date()} -> {X.index.max().date()}")
        print(f"Cibles de régression : {', '.join(cibles)}")

    X_train, X_test, _, _ = decoupe_temporelle(X, X.iloc[:, 0], annees_test)

    if verbose:
        print(f"Split temporel : train {len(X_train)} jours "
              f"({X_train.index.min().date()} -> {X_train.index.max().date()}) | "
              f"test {len(X_test)} jours ({X_test.index.min().date()} -> {X_test.index.max().date()})\n")

    date_entrainement = date.today().isoformat()
    commun = {
        "ville": ville, "lat": lat, "lon": lon,
        "features": features,
        "cibles_regression": list(cibles),
        "periode": {"debut": str(X.index.min().date()), "fin": str(X.index.max().date())},
        "split": {
            "train_fin": str(X_train.index.max().date()),
            "test_debut": str(X_test.index.min().date()),
            "annees_test": annees_test,
        },
        "date_entrainement": date_entrainement,
    }

    # ----- Régression : tmax, tmin, rhum, wspd, pres -----
    modeles_sauves: dict[str, str] = {}
    metriques_test: dict[str, dict] = {}
    metriques_train: dict[str, dict] = {}
    intervalles: dict[str, dict] = {}
    comparaisons: dict[str, pd.DataFrame] = {}
    predictions_test: dict[str, np.ndarray] = {}

    for cible, serie in cibles.items():
        if verbose:
            print(f"== Entraînement — {cible} (régression) ==")
        y_train = serie.loc[X_train.index]
        y_test = serie.loc[X_test.index]
        modele, nom, met_test, tableau = _entrainer_et_comparer(
            modeles_regression(), X_train, X_test, y_train, y_test, "regression", verbose
        )
        pred_test = modele.predict(X_test)
        metriques_test[cible] = met_test
        metriques_train[cible] = evaluer_regression(y_train, modele.predict(X_train))
        intervalles[cible] = intervalle_confiance(y_test, pred_test)
        comparaisons[cible] = tableau
        predictions_test[cible] = pred_test
        modeles_sauves[cible] = nom

        bundle = {
            **commun, "tache": "regression", "cible": cible,
            "nom_modele": nom, "model": modele,
            "metriques_test": met_test, "metriques_train": metriques_train[cible],
            "intervalle_confiance": intervalles[cible],
        }
        joblib.dump(bundle, os.path.join(MODELES_DIR, f"{cible}_{ville}.pkl"))

    # ----- Classification : risque de pluie -----
    if verbose:
        print("== Entraînement — Risque de pluie (classification) ==")
    y_train_rain = y_rain.loc[X_train.index]
    y_test_rain = y_rain.loc[X_test.index]
    modele_rain, nom_rain, met_rain, tab_rain = _entrainer_et_comparer(
        modeles_classification(), X_train, X_test, y_train_rain, y_test_rain, "classification", verbose
    )
    pred_rain_test = modele_rain.predict(X_test)
    proba_rain_test = modele_rain.predict_proba(X_test)[:, 1]
    metriques_test["rain"] = met_rain
    modeles_sauves["rain"] = nom_rain
    comparaisons["rain"] = tab_rain

    bundle_rain = {
        **commun, "tache": "classification", "cible": "rain",
        "nom_modele": nom_rain, "model": modele_rain,
        "seuil_pluie_mm": SEUIL_PLUIE_MM,
        "metriques_test": met_rain,
    }
    joblib.dump(bundle_rain, os.path.join(MODELES_DIR, f"rain_{ville}.pkl"))

    # ----- Prédictions du jeu de test (pour le dashboard) -----
    colonnes_test = {
        "tmax_reel": cibles["tmax"].loc[X_test.index] if "tmax" in cibles else np.nan,
        "tmax_pred": predictions_test.get("tmax", np.nan),
        "rain_reel": y_test_rain,
        "rain_pred": pred_rain_test,
        "rain_proba": proba_rain_test,
    }
    for cible, pred in predictions_test.items():
        colonnes_test[f"{cible}_pred"] = pred
        colonnes_test[f"{cible}_reel"] = cibles[cible].loc[X_test.index]
    test_preds = pd.DataFrame(colonnes_test, index=X_test.index)
    test_preds.index.name = "date"
    test_preds.to_csv(os.path.join(DATA_DIR, f"test_preds_{ville}.csv"), index_label="date")

    # ----- Métadonnées JSON (sans le modèle, lisible par un humain) -----
    infos = {
        **commun,
        "seuil_pluie_mm": SEUIL_PLUIE_MM,
        "niveau_intervalle": NIVEAU_INTERVALLE,
        "modeles": modeles_sauves,
        "intervalles": intervalles,
    }
    for cible in cibles:
        infos[cible] = {
            "nom_modele": modeles_sauves[cible],
            "metriques_test": metriques_test[cible],
            "metriques_train": metriques_train[cible],
            "intervalle_confiance": intervalles[cible],
        }
        infos[f"comparaison_{cible}"] = comparaisons[cible].reset_index().to_dict(orient="records")
    infos["rain"] = {"nom_modele": nom_rain, "metriques_test": met_rain}
    infos["comparaison_rain"] = tab_rain.reset_index().to_dict(orient="records")
    with open(os.path.join(MODELES_DIR, f"info_{ville}.json"), "w", encoding="utf-8") as f:
        json.dump(infos, f, ensure_ascii=False, indent=2)

    if verbose:
        print(f"Modèles sauvegardés dans models/ (ville = {ville}) : "
              f"{', '.join(f'{c}_{ville}.pkl' for c in CIBLES_REQUIS if c in modeles_sauves)}")
        print(f"Prédictions de test -> data/test_preds_{ville}.csv")

    return infos


def entrainer_pour_ville(
    ville: str,
    annees_test: int = ANNEES_TEST_DEFAUT,
    debut: str = "2010-01-01",
    fin: str | None = None,
    use_cache: bool = True,
    verbose: bool = True,
    lat: float | None = None,
    lon: float | None = None,
) -> dict:
    """Récupère les données d'une ville puis lance l'entraînement complet."""
    if lat is None or lon is None:
        ville = normaliser_ville(ville)
        if ville not in VILLES:
            raise KeyError(f"Ville inconnue : {ville}. Disponibles : {list(VILLES)}")
        lat, lon = VILLES[ville]
    fin = fin or (date.today() - timedelta(days=1)).isoformat()
    df = fetch_data(lat, lon, debut, fin, use_cache=use_cache)
    return entrainer_et_sauvegarder(df, ville, lat, lon, annees_test=annees_test, verbose=verbose)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entraînement des modèles météo")
    parser.add_argument("--ville", default="Sousse", help="Ville prédéfinie (ex. Sousse, Sidi Bouzid)")
    parser.add_argument("--lat", type=float, default=None, help="Latitude personnalisée")
    parser.add_argument("--lon", type=float, default=None, help="Longitude personnalisée")
    parser.add_argument("--debut", default="2010-01-01", help="Début de la période (AAAA-MM-JJ)")
    parser.add_argument("--fin", default=None, help="Fin de la période (défaut : hier)")
    parser.add_argument("--annees-test", type=int, default=ANNEES_TEST_DEFAUT,
                        help="Nombre d'années réservées au jeu de test")
    parser.add_argument("--no-cache", action="store_true", help="Ignore le cache CSV local")
    args = parser.parse_args()

    nom_ville = args.ville if args.lat is None else f"GPS_{args.lat:.2f}_{args.lon:.2f}"
    print(f"=== Entraînement — {nom_ville} ===\n")
    entrainer_pour_ville(
        nom_ville,
        annees_test=args.annees_test,
        debut=args.debut,
        fin=args.fin,
        use_cache=not args.no_cache,
        lat=args.lat,
        lon=args.lon,
    )
