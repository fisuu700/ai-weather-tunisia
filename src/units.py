"""
units.py
========
Support multi-unités de la plateforme.

Deux systèmes sont proposés dans l'interface :
  - « métrique »  : °C, km/h (unités natives des modèles) ;
  - « impérial »  : °F, mph.

Toutes les valeurs sont stockées et calculées en unités métriques ;
les conversions ne sont appliquées qu'à l'affichage (graphiques,
cartes, tableaux), ce qui garantit que les modèles restent inchangés.

Usage :
    from src import units as U
    U.fmt_temperature(26.4, U.IMPERIAL)   # '79.5 °F'
"""

from __future__ import annotations

METRIQUE = "metrique"
IMPERIAL = "imperial"

LIBELLES: dict[str, str] = {
    METRIQUE: "🌡️ Métrique (°C, km/h)",
    IMPERIAL: "🌡️ Impérial (°F, mph)",
}


def est_metric(systeme: str) -> bool:
    """Vrai si le système demandé est le système métrique."""
    return systeme != IMPERIAL


def temperature(celsius: float | None, systeme: str) -> float | None:
    """Convertit une température depuis les °C vers le système demandé."""
    if celsius is None:
        return None
    return float(celsius) if est_metric(systeme) else float(celsius) * 9.0 / 5.0 + 32.0


def delta_temperature(delta_celsius: float | None, systeme: str) -> float | None:
    """Convertit un ÉCART de température (offset sans référence zéro)."""
    if delta_celsius is None:
        return None
    return float(delta_celsius) if est_metric(systeme) else float(delta_celsius) * 9.0 / 5.0


def vitesse(kmh: float | None, systeme: str) -> float | None:
    """Convertit une vitesse depuis les km/h vers le système demandé."""
    if kmh is None:
        return None
    return float(kmh) if est_metric(systeme) else float(kmh) * 0.621371


def libelle_temperature(systeme: str) -> str:
    """Symbole d'unité de température (°C ou °F)."""
    return "°C" if est_metric(systeme) else "°F"


def libelle_vitesse(systeme: str) -> str:
    """Symbole d'unité de vitesse (km/h ou mph)."""
    return "km/h" if est_metric(systeme) else "mph"


def fmt_temperature(celsius: float | None, systeme: str, decimales: int = 1) -> str:
    """Formate une température avec son unité, ex. « 26.0 °C »."""
    val = temperature(celsius, systeme)
    if val is None:
        return "n/c"
    return f"{val:.{decimales}f} {libelle_temperature(systeme)}"


def fmt_delta_temperature(delta_celsius: float | None, systeme: str, decimales: int = 1) -> str:
    """Formate un écart de température signé, ex. « +1.8 °F »."""
    val = delta_temperature(delta_celsius, systeme)
    if val is None:
        return "n/c"
    return f"{val:+.{decimales}f} {libelle_temperature(systeme)}"


def fmt_vitesse(kmh: float | None, systeme: str, decimales: int = 0) -> str:
    """Formate une vitesse de vent avec son unité, ex. « 18 km/h »."""
    val = vitesse(kmh, systeme)
    if val is None:
        return "n/c"
    return f"{val:.{decimales}f} {libelle_vitesse(systeme)}"


def serie_temperature(values_celsius, systeme: str) -> list[float]:
    """Convertit une séquence de températures pour un graphique Plotly."""
    return [temperature(v, systeme) for v in values_celsius]


def serie_vitesse(values_kmh, systeme: str) -> list[float]:
    """Convertit une séquence de vitesses pour un graphique Plotly."""
    return [vitesse(v, systeme) for v in values_kmh]
