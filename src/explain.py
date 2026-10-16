"""
explain.py
==========
Explicabilité dynamique des prédictions (XAI — Explainable AI) via SHAP.

`explication_shap` calcule, pour UN jour précis, la contribution de
chaque feature à la prédiction du modèle (TreeExplainer, compatible
RandomForest / GradientBoosting / XGBoost) :

    prédiction = valeur de base (prior) + Σ contributions des features

Le module est tolérant à l'absence de `shap` : il renvoie simplement
`None`, et l'interface affiche un message explicatif à la place.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


def shap_disponible() -> bool:
    """Vrai si la bibliothèque shap est installée."""
    try:
        import shap  # noqa: F401
        return True
    except ImportError:
        return False


def _valeurs_plates(valeurs) -> np.ndarray:
    """
    Normalise la sortie de TreeExplainer.shap_values :
    tableau (n, p), (p,), liste de classes, ou (1, p, 1) selon la version.
    """
    if isinstance(valeurs, list):
        valeurs = valeurs[-1]
    tableau = np.asarray(valeurs, dtype=float)
    while tableau.ndim > 1:
        tableau = tableau[0]
    return tableau


def explication_shap(bundle: dict, X_row: pd.DataFrame) -> dict | None:
    """
    Explique localement la prédiction du modèle `bundle` pour la ligne
    `X_row` (une seule date, colonnes = features du modèle).

    Retour :
      {
        "base": valeur de base du modèle (prior sans les features),
        "valeurs": {feature: contribution signée, ...},
        "prediction": valeur prédite par le modèle pour cette ligne,
        "unite": "°C" pour une régression, "" pour une classification,
      }
      ou None si shap est absent ou si l'explication échoue.
    """
    try:
        import shap
    except ImportError:
        return None

    try:
        modele = bundle["model"]
        features = bundle["features"]
        X = X_row[features].fillna(0)

        explainer = shap.TreeExplainer(modele)
        valeurs = explainer.shap_values(X)
        plats = _valeurs_plates(valeurs)

        base = explainer.expected_value
        if isinstance(base, (list, np.ndarray)):
            base = np.asarray(base, dtype=float).ravel()[-1]

        tache = bundle.get("tache", "regression")
        if tache == "classification":
            prediction = float(modele.predict_proba(X)[0, 1])
            unite = ""
        else:
            prediction = float(modele.predict(X)[0])
            unite = "°C"

        return {
            "base": float(base),
            "valeurs": {f: float(v) for f, v in zip(features, plats)},
            "prediction": prediction,
            "unite": unite,
        }
    except Exception as exc:  # noqa: BLE001 — toute erreur shap dégrade l'affichage
        warnings.warn(f"Explication SHAP indisponible : {exc}")
        return None
