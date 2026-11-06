"""
src/tounsi_ai.py
================
Module d'Intelligence Artificielle Météo en dialecte tunisien (Derja / تونسية).
Comprend le langage tunisien à 100% (Arabizi / Franco-arabe et texte arabe)
et répond en authentique dialecte tunisien avec des données météorologiques réelles.
"""

from __future__ import annotations

import os
import re
import unicodedata
from datetime import date, datetime, timedelta
from typing import Any

from src.data_loader import VILLES, normaliser_ville, obtenir_info_geographique
from src import predict as predict_module
from src import weather_api


# ==============================================================================
# Vocabulaire & Dictionnaires Tunisiens (Arabizi + Arabe)
# ==============================================================================

MAPPING_VILLES_TUNISIENNES = {
    # Grand Tunis
    "tunis": "Tunis", "tounes": "Tunis", "tunes": "Tunis", "تونس": "Tunis",
    "ariana": "Ariana", "l'ariana": "Ariana", "aryana": "Ariana", "أريانة": "Ariana",
    "ben arous": "Ben Arous", "benarous": "Ben Arous", "بن عروس": "Ben Arous",
    "manouba": "Manouba", "la manouba": "Manouba", "mnouba": "Manouba", "منوبة": "Manouba",
    # Nord
    "bizerte": "Bizerte", "benzart": "Bizerte", "binzart": "Bizerte", "بنزرت": "Bizerte",
    "nabeul": "Nabeul", "nabel": "Nabeul", "nebeul": "Nabeul", "نابل": "Nabeul",
    "zaghouan": "Zaghouan", "zaghwen": "Zaghouan", "زغوان": "Zaghouan",
    "beja": "Béja", "béja": "Béja", "baja": "Béja", "باجة": "Béja",
    "jendouba": "Jendouba", "jendouba": "Jendouba", "jandouba": "Jendouba", "جندوبة": "Jendouba",
    "kef": "Le Kef", "le kef": "Le Kef", "el kef": "Le Kef", "elkef": "Le Kef", "الكاف": "Le Kef",
    "siliana": "Siliana", "silyana": "Siliana", "سليانة": "Siliana",
    # Centre
    "sousse": "Sousse", "soussa": "Sousse", "سوسة": "Sousse",
    "monastir": "Monastir", "mestir": "Monastir", "mostir": "Monastir", "المنستير": "Monastir",
    "mahdia": "Mahdia", "mehdia": "Mahdia", "المهدية": "Mahdia",
    "sfax": "Sfax", "sfa9es": "Sfax", "sfaques": "Sfax", "صفاقس": "Sfax",
    "kairouan": "Kairouan", "9ayrawan": "Kairouan", "qayrawan": "Kairouan", "القيروان": "Kairouan",
    "kasserine": "Kasserine", "gasrine": "Kasserine", "gasrin": "Kasserine", "القصرين": "Kasserine",
    "sidi bouzid": "Sidi Bouzid", "sidibouzid": "Sidi Bouzid", "bouzid": "Sidi Bouzid", "سيدي بوزيد": "Sidi Bouzid",
    # Sud
    "gabes": "Gabès", "gabès": "Gabès", "gabes": "Gabès", "gabis": "Gabès", "قابس": "Gabès",
    "medenine": "Médenine", "médenine": "Médenine", "mednin": "Médenine", "djerba": "Médenine", "جرجيس": "Médenine", "مدنين": "Médenine",
    "tataouine": "Tataouine", "tatwin": "Tataouine", "tatawin": "Tataouine", "تطاوين": "Tataouine",
    "gafsa": "Gafsa", "gafsa": "Gafsa", "gaf9a": "Gafsa", "قفصة": "Gafsa",
    "tozeur": "Tozeur", "tewzer": "Tozeur", "touzer": "Tozeur", "توزر": "Tozeur",
    "kebili": "Kébili", "kébili": "Kébili", "gbeli": "Kébili", "gbelli": "Kébili", "قبلي": "Kébili",
}

MOTS_CLES_PLUIE = [
    "mtar", "matar", "nouwa", "nwa", "chita", "chta", "tboul", "moussem", "pluie", "parapluie",
    "madhalla", "tsob", "tsab", "tsaksek", "مطاف", "مطر", "نوة", "شتاء", "تصب"
]

MOTS_CLES_FROID = [
    "bard", "berd", "bareda", "bared", "zmigri", "garsa", "glacon", "froid", "برودة", "برد"
]

MOTS_CLES_CHAUD = [
    "s5ana", "skhana", "s5oun", "skhoun", "chams", "chems", "9ayla", "gayla", "chhar kawn",
    "chaouba", "chawba", "chaud", "chaleur", "سخانة", "سخون", "شمس", "قائلة"
]

MOTS_CLES_VENT = [
    "rya7", "ryeh", "rih", "3ejja", "3ajja", "ghbar", "vent", "souffle", "ريح", "عجاج", "رياح"
]

MOTS_CLES_MER = [
    "b7ar", "bhar", "ba7r", "3wem", "3wema", "3oum", "mouj", "mer", "baignade", "plage",
    "بحر", "عوم", "عومة", "موج"
]

MOTS_CLES_VETEMENTS = [
    "nelbes", "nalbes", "lebsa", "labsa", "debch", "veste", "kabbout", "manteau", "tshirt",
    "short", "chnebes", "nhezz", "nhez", "شلبس", "نلبس", "دبش", "كسوة", "لبسة"
]

MOTS_CLES_SORTIE = [
    "5rouj", "khrouj", "nokhroj", "no5roj", "9ahwa", "9ahwet", "sefra", "tsafira", "footing",
    "dar", "machya", "sortie", "نخرج", "خروج", "قهوة", "سفر"
]

MOTS_CLES_SALUTATION = [
    "3aslema", "aslema", "ahla", "ahla bik", "ahla w sahla", "mar7ba", "marhba", "salut",
    "salam", "bonjour", "bonsoir", "coucou", "sbah el khir", "sbe7 el khir", "sbe7 el fol",
    "مرحبا", "عسلامة", "أهلا", "سلام", "صباح الخير"
]

MOTS_CLES_ETAT = [
    "chna7welek", "chne7welek", "chnowa a7welek", "a7welek", "cv", "ca va", "ça va",
    "labes", "lebes", "kifech 7alek", "kif 7alek", "chfama jdid", "kifech el a7wel",
    "شنحوالك", "شحوالك", "لاباس", "كيف حالك"
]

MOTS_CLES_REMERCIEMENT = [
    "merci", "chokran", "chokran jazilan", "3aychek", "ya3tik essa7a", "ya3tik sa7a",
    "sahhit", "sa7it", "barak allah fik", "thx", "thanks",
    "شكرا", "يعيشك", "يعطيك الصحة", "صحيت", "بارك الله فيك"
]

MOTS_CLES_IDENTITE = [
    "chkoun enti", "chkounik", "chneya enti", "chnia enti", "chbik", "qui es tu", "qui es-tu",
    "tu es qui", "c'est quoi", "presente toi", "présente-toi", "chta3mel",
    "شكون أنت", "من أنت", "عرف بنفسك"
]

MOTS_CLES_AU_REVOIR = [
    "besslema", "beslema", "bye", "au revoir", "ciao", "bonne nuit", "lila sa3ida",
    "tchao", "besslama", "ila li9aa", "بسلامة", "إلى اللقاء", "تصبح على خير"
]

MOTS_CLES_ACCORD = [
    "mrigel", "behi", "bahi", "ok", "daccord", "d'accord", "c bon", "sa7a", "wadha7",
    "واضح", "باهي", "مريقل"
]


def _nettoyer_texte(txt: str) -> str:
    """Normalise et nettoie le texte pour l'analyse lexicale."""
    txt = txt.lower().strip()
    txt = unicodedata.normalize("NFD", txt)
    txt = "".join(c for c in txt if unicodedata.category(c) != "Mn")
    return txt


def detecter_gouvernorat(texte: str, gouvernorat_par_defaut: str = "Tunis") -> str:
    """Détecte si un gouvernorat tunisien est mentionné dans la question."""
    texte_clean = _nettoyer_texte(texte)
    # Chercher d'abord les mots de plus de 3 lettres
    for mot, gouv in sorted(MAPPING_VILLES_TUNISIENNES.items(), key=lambda x: len(x[0]), reverse=True):
        pattern = r"\b" + re.escape(mot) + r"\b"
        if re.search(pattern, texte_clean):
            return gouv
    return gouvernorat_par_defaut


def detecter_horizon(texte: str) -> tuple[int, str]:
    """Détecte l'horizon temporel demandé (0 = aujourd'hui/taw, 1 = ghodwa, 2 = ba3d ghodwa)."""
    t = _nettoyer_texte(texte)
    if "ba3d ghodwa" in t or "baad ghodwa" in t or "après-demain" in t:
        return 2, "ba3d ghodwa"
    if "ghodwa" in t or "demain" in t or "غدوة" in t:
        return 1, "ghodwa"
    if "weekend" in t or "jem3a hedhi" in t or "el jem3a" in t:
        return 3, "el ayamet ejjeyin"
    return 0, "lyoum"


def detecter_question_geographie(texte: str) -> bool:
    """Détecte si la question interroge la position, la situation ou le lieu géographique d'une ville."""
    t = _nettoyer_texte(texte)

    # Termes explicites de géographie et de situation
    mots_cles = [
        "place geographique", "position geographique", "situation geographique",
        "ou se trouve", "ou elle se trouve", "ou il se trouve", "ou est", "ou se situe",
        "localisation", "geographie", "joghrafya", "joghrafi", "maw9e3", "mow9e3",
        "mow9a3", "maw9a3", "7oudoud", "jiran", "frontiere",
        "موقع", "وين جات", "وين تقع", "وين بلاصتها", "موقعها", "جغرافيا", "أين تقع", "بلاصة"
    ]
    if any(k in t for k in mots_cles):
        return True

    # Formules en Derja : "win jet", "win jéya", "win mawjouda", "win to93od", "fin jet", "winha"
    if re.search(r"\b(win|wayn|fin|feen)\s+(jet|jeya|jey|mawjouda|to93od|to93ad|blaset|blasa|blasetha)\b", t):
        return True

    # "win <ville>" ou "<ville> win" ou "winha <ville>" (ex: "win beja", "tozeur win", "winha sousse")
    for ville_cle in MAPPING_VILLES_TUNISIENNES:
        p1 = r"\b(win|wayn|fin|feen)\s+" + re.escape(ville_cle) + r"\b"
        p2 = r"\b" + re.escape(ville_cle) + r"\s+(win|wayn|fin|feen)\b"
        p3 = r"\b(winha|waynha)\s+" + re.escape(ville_cle) + r"\b"
        if re.search(p1, t) or re.search(p2, t) or re.search(p3, t):
            return True

    # Si les mots "place" ou "region" ou "région" sont associés à un nom de ville
    if any(w in t for w in ["place", "region", "région"]) and any(v in t for v in MAPPING_VILLES_TUNISIENNES):
        return True

    return False


def contient_mot(texte: str, mots: list[str]) -> bool:
    """Vérifie si au moins un des mots apparaît en tant que mot entier."""
    for m in mots:
        pattern = r"(?<!\w)" + re.escape(m) + r"(?!\w)"
        if re.search(pattern, texte):
            return True
    return False


def est_message_purement_conversationnel(texte_clean: str) -> tuple[bool, str]:
    """
    Détecte si le message de l'utilisateur est une salutation, un remerciement,
    une formule de politesse ou une question sur l'état sans demande météo.
    Retourne (est_conversationnel, intention).
    """
    # Identité en premier ("chkoun enti", "qui es tu")
    if contient_mot(texte_clean, MOTS_CLES_IDENTITE):
        return True, "identite"

    # Remerciements ("merci", "chokran", "3aychek")
    if contient_mot(texte_clean, MOTS_CLES_REMERCIEMENT):
        return True, "merci"

    # Au revoir ("besslema", "bye", "au revoir")
    if contient_mot(texte_clean, MOTS_CLES_AU_REVOIR):
        return True, "au_revoir"

    # État / Prendre des nouvelles ("chna7welek", "cv", "labes")
    if contient_mot(texte_clean, MOTS_CLES_ETAT):
        return True, "etat"

    # Mots-clés indiquant une vraie question météo ou territoriale
    indices_meteo = (
        MOTS_CLES_PLUIE + MOTS_CLES_FROID + MOTS_CLES_CHAUD + MOTS_CLES_VENT +
        MOTS_CLES_MER + MOTS_CLES_VETEMENTS + MOTS_CLES_SORTIE + [
            "ta9s", "taqs", "te9s", "meteo", "météo", "jaw", "jawha", "climat", "7arara", "harara",
            "darajat", "temperature", "température", "degre", "degré", "ghodwa", "demain",
            "lyoum", "ba3d ghodwa", "weekend", "chams", "s7ab", "طقس", "جو", "حرارة", "غدوة", "مطر"
        ]
    )
    # S'il y a un mot météo ou une question géographique, ce n'est pas purement conversationnel
    if contient_mot(texte_clean, indices_meteo):
        return False, ""
    if detecter_question_geographie(texte_clean):
        return False, ""

    # Salutation pure ("ahla", "3aslema", "salut", "bonjour")
    if contient_mot(texte_clean, MOTS_CLES_SALUTATION):
        return True, "salutation"

    # Accord / Confirmation ("mrigel", "behi", "ok")
    if contient_mot(texte_clean, MOTS_CLES_ACCORD):
        return True, "accord"

    return False, ""


# ==============================================================================
# Récupérateur de données météo & géographiques enrichies pour l'IA
# ==============================================================================

def recuperer_contexte_meteo(ville: str) -> dict[str, Any]:
    """Récupère les observations en direct, prévisions prédictives et géographie du gouvernorat."""
    ville_norm = normaliser_ville(ville)
    lat, lon = VILLES.get(ville_norm, VILLES["Tunis"])
    info_geo = obtenir_info_geographique(ville_norm)

    contexte = {
        "ville": ville_norm,
        "lat": lat,
        "lon": lon,
        "geo": info_geo,
        "live": None,
        "previsions": [],
        "source": "Open-Meteo & ML Models",
    }

    try:
        live = weather_api.conditions_actuelles(lat, lon)
        contexte["live"] = live
    except Exception:
        contexte["live"] = None

    try:
        prev = predict_module.predire_multijours(ville=ville_norm, lat=lat, lon=lon, jours=5)
        contexte["previsions"] = prev.get("previsions", [])
    except Exception:
        # Fallback API météo standard
        try:
            api_prev = weather_api.prevision_journaliere(lat, lon, jours=5)
            if not api_prev.empty:
                for idx, row in api_prev.iterrows():
                    contexte["previsions"].append({
                        "date": str(idx.date()) if hasattr(idx, "date") else str(idx),
                        "tmax": row.get("tmax", 25.0),
                        "tmin": row.get("tmin", 16.0),
                        "proba_pluie": float(row.get("prcp", 0.0) > 1.0) * 0.4,
                        "wspd": row.get("wspd", 15.0),
                        "rhum": row.get("rhum", 65.0),
                        "pres": 1015.0,
                    })
        except Exception:
            contexte["previsions"] = []

    return contexte


# ==============================================================================
# Moteur IA en dialecte tunisien (Derja) natif et ultra-intelligent
# ==============================================================================

def repondre_en_tunisien_moteur_local(question: str, contexte: dict[str, Any]) -> str:
    """Génère une réponse complète, fluide et authentique en derja tunisienne."""
    q_clean = _nettoyer_texte(question)
    ville = contexte["ville"]

    # 1. Vérification si le message est purement conversationnel (salutation, politesse, etc.)
    est_conv, type_conv = est_message_purement_conversationnel(q_clean)
    if est_conv:
        if type_conv == "salutation":
            saluts_purs = [
                f"Ahla w sahla bik ! Y3aychek ya ghali 😊 Mar7ba bik 🇹🇳.\n\nEna m3ak bech n3awnek f kol chay y5os ta9s tounes. Chnowa t7eb ta3ref elyoum ? Tnajjem tes'elni 3la ta9s {ville}, win jéya ay blasa, el mtar, el 7arara wela chnowa telbes !",
                f"3aslema bik ya sidi ! Ahla w mar7ba 🇹🇳.\n\nNchallah nharak tayeb ! Ena el assistant météo mte3ek, m3ak taw f **{ville}**. Chnowa nchouflek : ta9s lyoum, ghodwa, mtar wela blasa o5ra ?",
                f"Ahla bik ! Mar7ba bik f ay wa9t 🇹🇳.\n\nKifech nnejjem n3awnek elyoum ? As'elni 3la ta9s ay gouvernorat f tounes (7arara, mtar, rya7, b7ar) w taw njewbek toul !",
            ]
            return saluts_purs[hash(question) % len(saluts_purs)]
        elif type_conv == "etat":
            etats = [
                f"Hamdoulilah ya rabi, kol chay mrigel w 3al el 3al ! 🌤️ Y3aychek 3al sou2el.\nW enti chna7welek, nchallah labes ?\n\nChnowa t7eb ta3ref 3la ta9s elyoum wela el gouvernorats ?",
                f"Labes l'7amdoullah, kol chay behi ! 😊 Nchallah enti zeda labes w b khir.\n\nEna houni f khedmtek ken 7achtek b ay ma3louma 3la ta9s {ville} wela ay blasa f tounes !",
            ]
            return etats[hash(question) % len(etats)]
        elif type_conv == "identite":
            return (
                f"Ena **Tounsi Météo AI** (مساعد الطقس التونسي الذكي) 🇹🇳 !\n\n"
                f"Assistant dhiqui makhdoum bech yafhem el tounsi 100% (Derja w Arabizi). "
                f"Na3ref ta9s el 24 gouvernorats kolha b les données réelles (7arara, mtar, rya7, b7ar, chnowa telbes), "
                f"w na3ref el maw9e3 el joghrafi mte3 kol wleyah.\n\n"
                f"Taw a7na f gouvernorat **{ville}**. As'elni b tounsi kima t7eb w njewbek !"
            )
        elif type_conv == "merci":
            return f"Y3aychek ya ghali ! La chokr 3la wajeb 😊. Ena dima ba7dhek ken st7a9it ay ma3louma 3al météo wela ta9s tounes !"
        elif type_conv == "au_revoir":
            return f"Besslema w fi amani Lah ! Nharak tayeb w rod balek 3la rou7ek 👋🇹🇳. Mar7ba bik fi ay wa9t !"
        elif type_conv == "accord":
            return f"Mrigel ya sidi ! Ken 7achtek b ay 7aja o5ra fel météo, ena houni m3ak 😊."

    live = contexte.get("live")
    previsions = contexte.get("previsions", [])

    # Températures de base
    temp_actuelle = live.get("temp_c") if live else None
    meteo_desc = live.get("description", "").lower() if live else ""
    vent_actuel = live.get("wind_kmh") if live else None
    hum_actuelle = live.get("humidity_pct") if live else None

    # Données prévision
    horizon_idx, horizon_nom = detecter_horizon(question)
    prev_cible = None
    if previsions:
        idx_reel = min(horizon_idx, len(previsions) - 1)
        prev_cible = previsions[idx_reel]

    tmax = prev_cible["tmax"] if prev_cible else (temp_actuelle or 26.0)
    tmin = prev_cible["tmin"] if prev_cible else ((temp_actuelle or 20.0) - 8.0)
    proba_pluie = (prev_cible.get("proba_pluie", 0.0) * 100) if prev_cible else 10.0
    vent = prev_cible.get("wspd", vent_actuel or 18.0) if prev_cible else (vent_actuel or 18.0)
    hum = prev_cible.get("rhum", hum_actuelle or 60.0) if prev_cible else (hum_actuelle or 60.0)

    # Détection de l'intention de la question
    a_geographie = detecter_question_geographie(question)
    a_pluie = contient_mot(q_clean, MOTS_CLES_PLUIE)
    a_froid = contient_mot(q_clean, MOTS_CLES_FROID)
    a_chaud = contient_mot(q_clean, MOTS_CLES_CHAUD)
    a_vent = contient_mot(q_clean, MOTS_CLES_VENT)
    a_mer = contient_mot(q_clean, MOTS_CLES_MER)
    a_vetements = contient_mot(q_clean, MOTS_CLES_VETEMENTS)
    a_sortie = contient_mot(q_clean, MOTS_CLES_SORTIE)
    a_salut = contient_mot(q_clean, MOTS_CLES_SALUTATION)

    reponse_parts = []

    # Salutation amicale en préambule d'une réponse météo
    if a_salut and not a_geographie:
        saluts = [
            f"3aslema bik ya sidi ! Mar7ba bik f météo {ville} 🇹🇳.",
            f"Ahla w sahla bik ! Y3aychek, a7welek labes ?",
            f"3aslema ! Ena m3ak bech n9ollek kol chay 3la ta9s f {ville} !",
        ]
        reponse_parts.append(saluts[hash(question) % len(saluts)])

    # 0. QUESTION SUR LA SITUATION GÉOGRAPHIQUE / MAW9E3 EL JOGHRAFI
    if a_geographie:
        geo = contexte.get("geo") or obtenir_info_geographique(ville)
        nom_ar = geo.get("nom_ar", ville)
        region = geo.get("region", "Tunisie")
        zone = geo.get("zone", "")
        milieu = geo.get("milieu", "")
        altitude = geo.get("altitude_m", 0)
        voisins = geo.get("voisins", [])
        voisins_str = ", ".join(voisins) if voisins else "Manti9a dakhiliya"
        desc_derja = geo.get("description_derja", "")
        desc_fr = geo.get("description_fr", "")
        climat_geo = geo.get("climat", "")
        facade = geo.get("facade_maritime", False)

        mer_str = "🌊 **El ba7r :** 3andha chwate2 w wajha ba7riya (Côtière / Sahiliya)." if facade else "🏞️ **El ba7r :** Gouvernorat dakhili (Continentale / ma fehech b7ar direct)."

        reponse_parts.append(
            f"🗺️ **El maw9e3 el joghrafi mte3 {ville} ({nom_ar}) f Tounes 🇹🇳 :**\n\n"
            f"- 🧭 **El jiha wel région :** **{region}** (f {zone} el tounsi).\n"
            f"- 📍 **Win jéya bedhabt :** {desc_derja}\n"
            f"- 🏞️ **El tabi3a wel tadhariis :** {milieu} (3louha 9rib **{altitude} m** 3al b7ar).\n"
            f"- {mer_str}\n"
            f"- 🤝 **El jiran wel 7oudoud :** Dayer biha les gouvernorats : **{voisins_str}**.\n"
            f"- 📌 **Les coordonnées GPS :** Latitude `{geo.get('lat', 0.0):.4f}°N`, Longitude `{geo.get('lon', 0.0):.4f}°E`.\n\n"
            f"🌦️ **Ta'thir el joghrafya 3al ta9s :**\n"
            f"*{climat_geo}*"
        )
        if temp_actuelle is not None:
            reponse_parts.append(
                f"🌡️ **Ta9s taw fi {ville} :** Fama **{temp_actuelle:.1f}°C** ({meteo_desc}), "
                f"Tmax elyoum **{tmax:.1f}°C**, rya7 **{vent:.0f} km/h**."
            )

    # 1. QUESTION SUR LA PLUIE / MTAR
    elif a_pluie:
        if proba_pluie >= 50:
            reponse_parts.append(
                f"🌧️ **Ey fama mtar w nouwa f {ville} {horizon_nom} !**\n"
                f"El risque mte3 el mtar wesel l **{proba_pluie:.0f}%** avec des averses probables. "
                f"A3mel 7sabek w 5alli el madhalla (parapluie) dima m3ak f ydek w rod balek mil glissade f thniya !"
            )
        elif proba_pluie >= 25:
            reponse_parts.append(
                f"🌦️ **Fama chwaya risque mte3 mtar 5fifa (nadh7a) f {ville} {horizon_nom} ({proba_pluie:.0f}%)**.\n"
                f"El sma tnajjem tkoun mghayma chwaya b s7ab, ama ma famech tbe7be7 kbir. Hezz m3ak veste wela parapluie juste pour la sécurité."
            )
        else:
            reponse_parts.append(
                f"☀️ **Laa, ma famech mtar f {ville} {horizon_nom} !**\n"
                f"El risque mte3 el mtar d3if barcha (**{proba_pluie:.0f}%** barka). "
                f"El jaw sa7i w mrigel, tnajjem to5roj w t7awwes labes 3lik !"
            )

    # 2. QUESTION SUR LE FROID / S5ANA
    elif a_froid or a_chaud:
        if tmax >= 32:
            reponse_parts.append(
                f"🔥 **Ta9s s5oun barcha f {ville} {horizon_nom} !**\n"
                f"El 7arara bch tousel l **{tmax:.1f}°C** (w tmin **{tmin:.1f}°C**). "
                f"Chams tachwa9 w qayla 9wiya ! Rod balek mil chams, ochrob barcha meee w efda el 5rouj wa9t el 9ayla."
            )
        elif tmax >= 24:
            reponse_parts.append(
                f"🌤️ **Jaw dabouza w mo3tadel (cha3bi w ta7foun) f {ville} {horizon_nom} !**\n"
                f"El 7arara el 9oswa fi 7oudoud **{tmax:.1f}°C** wel soufla **{tmin:.1f}°C**. "
                f"Ta9s mte3 5rouj w 9ahwa 3la barra, la houwa bard la houwa s5oun barcha."
            )
        elif tmax >= 16:
            reponse_parts.append(
                f"🍂 **Fama chwaya zmigri w bard 5fif f {ville} {horizon_nom} !**\n"
                f"Tmax bch tkoun **{tmax:.1f}°C** w fel lil / sbe7 el bekri thabbet l **{tmin:.1f}°C**. "
                f"Albes 7aja tdhafik (veste behya) 5ater t7ess b nesma barda."
            )
        else:
            reponse_parts.append(
                f"❄️ **Bard w zmigri s7i7 f {ville} {horizon_nom} !**\n"
                f"El 7arara ma tfoutech **{tmax:.1f}°C** w tousel l **{tmin:.1f}°C** fel lil ! "
                f"Tdaffa mli7, albes pull s5oun w kabbout w ochrob 7aja s5ouna !"
            )

    # 3. QUESTION SUR LES VÊTEMENTS (CHNOWA NELBES)
    elif a_vetements:
        reponse_parts.append(f"👕 **Chnowa telbes {horizon_nom} f {ville} :**")
        if tmax >= 30:
            reponse_parts.append(
                f"Ma dem el 7arara bch tousel l **{tmax:.1f}°C**, albes 7weyej 9tan 5fifa (t-shirt w short/serwel 5fif), "
                f"chapeau w lunettes de soleil 🕶️, w ochrob el me mli7 !"
            )
        elif tmax >= 22:
            reponse_parts.append(
                f"El 7arara **{tmax:.1f}°C** fel 9ayla ama **{tmin:.1f}°C** fel lil w sbe7. "
                f"Albes 7aja demi-saison (t-shirt m3ah tricot 5fif wela gilet) bech ken berd el jaw fel lil telbes veste 🧥."
            )
        else:
            reponse_parts.append(
                f"El ta9s bared (Tmax **{tmax:.1f}°C** / Tmin **{tmin:.1f}°C**). "
                f"Albes kabbout wela veste s5ouna mli7 w foulard, w kenik 5arej f lil dharba bard !"
            )
        if proba_pluie >= 35:
            reponse_parts.append(f"⚠️ **W matensach el madhalla (parapluie)** 5ater fama risque mtar ({proba_pluie:.0f}%) !")

    # 4. QUESTION SUR LA MER / BAIGNADE (B7AR)
    elif a_mer:
        geo = contexte.get("geo") or obtenir_info_geographique(ville)
        if not geo.get("facade_maritime", False):
            reponse_parts.append(
                f"🏞️ **Ya sidi, {ville} ({geo.get('region')}) manti9a dakhiliya ma fehach b7ar mubachir !**\n"
                f"Ken nawi 3al 3wema w chott, a9reb chwate2 lik houma fel Sahel (Sousse/Monastir/Mahdia), "
                f"Cap Bon (Nabeul/Hammamet), Bizerte wela Gabès/Djerba selon blastek.\n"
                f"Ta9s {ville} {horizon_nom} : 7arara **{tmax:.1f}°C**, rya7 **{vent:.0f} km/h**."
            )
        elif vent >= 30:
            reponse_parts.append(
                f"🌊 **El ba7r mhayej w mouch mte3 3wem f {ville} !**\n"
                f"El rya7 tousel l **{vent:.0f} km/h** m3a amwej 3alya. "
                f"Men a7sen teb3ed 3al 3wem w ta3mel 9ahwa ba7dha el ba7r barka 3la safety !"
            )
        elif tmax < 22:
            reponse_parts.append(
                f"🥶 **El me bared barcha 3al 3wema f {ville} !**\n"
                f"Tmax **{tmax:.1f}°C**, el jaw mouch mte3 b7ar w 3wem, ama ta3mel dwicha wela tmechya 3al rmal jawha béhi."
            )
        else:
            reponse_parts.append(
                f"🏊 **Jaw ba7r 3alami f {ville} !**\n"
                f"El rya7 rathya (**{vent:.0f} km/h**) wel 7arara **{tmax:.1f}°C**. "
                f"El ba7r hani w rekadh, chwaya crême solaire w a3mel ghtsa se7liya !"
            )

    # 5. QUESTION SUR LE VENT (RYA7)
    elif a_vent:
        if vent >= 35:
            reponse_parts.append(
                f"💨 **Fama rya7 9wiya barcha f {ville} {horizon_nom} !**\n"
                f"Sor3et el rya7 bch tousel l **{vent:.0f} km/h**. "
                f"Rod balek wenti soug f el kayes (surtout autoroute) w stanna chwaya 3ejja kenek fel jihet el da5iliya."
            )
        else:
            reponse_parts.append(
                f"🍃 **El rya7 hanna w d3ifa f {ville} {horizon_nom} (**{vent:.0f} km/h**)**.\n"
                f"Nesma dhyefa w ma thamech 9la9 jemla fil dhourouf hedhi."
            )

    # 6. QUESTION SUR SORTIE / JAW GENERAL
    elif a_sortie:
        if proba_pluie < 30 and vent < 28 and 18 <= tmax <= 32:
            reponse_parts.append(
                f"✨ **Jaw 5rouj 5amsa 3la 5amsa f {ville} {horizon_nom} !**\n"
                f"El ta9s sa7i, 7arara **{tmax:.1f}°C**, w rya7 **{vent:.0f} km/h**.\n"
                f"5arej l'famille wela a3mel 9ahwa f terrasse, el ta9s yfatte9 !"
            )
        else:
            reponse_parts.append(
                f"⚠️ **Ken bch to5roj {horizon_nom} f {ville}, rod balek :**\n"
                f"Tmax: **{tmax:.1f}°C**, Risque mtar: **{proba_pluie:.0f}%**, Vent: **{vent:.0f} km/h**.\n"
                f"Ken fama mtar wela rya7, men a7sen 9a3da fe dar wela blasa mgatya."
            )

    # 7. RAPPORT GENERAL (PAR DEFAUT)
    else:
        live_str = f"Taw fi {ville} fama **{temp_actuelle:.1f}°C** ({meteo_desc})." if temp_actuelle else ""
        reponse_parts.append(
            f"🌤️ **Hak résumé ta9s mte3 {ville} {horizon_nom} :**\n\n"
            f"{live_str}\n"
            f"- 🌡️ **El 7arara el 9oswa (Tmax) :** **{tmax:.1f}°C**\n"
            f"- 🌡️ **El 7arara el soufla (Tmin) :** **{tmin:.1f}°C**\n"
            f"- 🌧️ **Risque el mtar :** **{proba_pluie:.0f}%** ({'Fama chwaya mtar' if proba_pluie >= 35 else 'Sa7i w chams'})\n"
            f"- 💨 **El rya7 :** **{vent:.0f} km/h**\n"
            f"- 💧 **El rtouba (Humidité) :** **{hum:.0f}%**\n\n"
            f"💡 *Ken t7eb tes'el 3la gouvernorat a5er (kima Sfax, Sousse, Bizerte...) wela chnowa telbes, 9olli taw njewbek !*"
        )

    return "\n\n".join(reponse_parts)


# ==============================================================================
# Interface Principale : Assistant Hybride (OpenAI si clé dispo + Moteur Derja)
# ==============================================================================

def generer_reponse_ai_tounsi(
    question: str,
    historique: list[dict[str, str]],
    gouvernorat_actif: str = "Tunis",
    api_key: str | None = None,
) -> str:
    """
    Point d'entrée principal :
    1. Détecte le gouvernorat mentionné (ou utilise le gouvernorat actif)
    2. Récupère les vraies données météo
    3. Tente OpenAI (si clé valide fournie) avec un prompt tunisien expert
    4. Bascule de façon transparente sur le moteur tunisien local sinon.
    """
    # Détection du gouvernorat ciblé dans le message
    ville_cible = detecter_gouvernorat(question, gouvernorat_par_defaut=gouvernorat_actif)
    contexte = recuperer_contexte_meteo(ville_cible)

    cle_openai = api_key or os.environ.get("OPENAI_API_KEY")

    # Si une clé OpenAI semble être valide (pas le placeholder)
    if cle_openai and not cle_openai.startswith("your_api") and len(cle_openai) > 20:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=cle_openai)

            geo = contexte.get("geo") or obtenir_info_geographique(ville_cible)
            resume_meteo = f"""
Données géographiques réelles pour {ville_cible} ({geo.get('nom_ar')}):
- Région administrative: {geo.get('region')} ({geo.get('zone')} de la Tunisie)
- Milieu naturel / Relief: {geo.get('milieu')} (Altitude: ~{geo.get('altitude_m')} m)
- Façade maritime: {'Oui (Côte méditerranéenne)' if geo.get('facade_maritime') else 'Non (Gouvernorat intérieur continental)'}
- Gouvernorats voisins: {', '.join(geo.get('voisins', []))}
- Situation géographique exacte: {geo.get('description_fr')}
- En Derja tunisienne: {geo.get('description_derja')}
- Climat local: {geo.get('climat')}
- Coordonnées GPS: Latitude {geo.get('lat')}°N, Longitude {geo.get('lon')}°E

Données météo réelles pour {ville_cible} (Tunisie):
- Température actuelle (Live): {contexte.get('live', {}).get('temp_c', 'N/A')} °C
- Description actuelle: {contexte.get('live', {}).get('description', 'N/A')}
- Vent actuel: {contexte.get('live', {}).get('wind_kmh', 'N/A')} km/h
- Prévisions prochainement:
"""
            for p in contexte.get("previsions", [])[:3]:
                resume_meteo += f"  * Date {p.get('date')}: Tmax {p.get('tmax'):.1f}°C, Tmin {p.get('tmin'):.1f}°C, Risque pluie {p.get('proba_pluie', 0)*100:.0f}%, Vent {p.get('wspd'):.0f} km/h\n"

            system_prompt = f"""
Enti "Tounsi Météo AI" (مساعد الطقس التونسي الذكي), assistant météo expert 100% tunisien.
Tu parles couramment et authentiquement en dialecte tunisien (Derja en alphabet latin Arabizi / Franco-arabe et quelques mots en arabe).
Tu comprends parfaitement le tunisien (argot, expressions météo : s5ana, zmigri, bard, mtar, rya7, b7ar, qayla, madhalla, chnowa nelbes, jaw el 5rouj).
Tu connais parfaitement la place géographique des 24 gouvernorats tunisiens (nord, sud, sahel, tell, oasis, désert, montagnes, côtes) et leur climat.
Utilise toujours les données météorologiques et géographiques réelles fournies ci-dessous pour être ultra précis et utile:
{resume_meteo}
Gouvernorat ciblé: {ville_cible}.
Règles :
1. Réponds toujours chaleureusement en tunisien (Derja naturelle, fluide, ni trop formelle ni vulgaire).
2. Si on te demande où se trouve une ville / sa position géographique ("win jet", "fin jéya", "place géographique", "où se trouve"), décris sa place géographique avec précision en tunisien (région, voisins, mer ou intérieur, particularité territoriale).
3. Donne des conseils pratiques et utiles (vêtements à porter, baignade, parapluie, conduite) adaptés à la réalité géographique du gouvernorat.
4. Sois concis, clair et sympathique avec des emojis météo.
"""
            messages = [{"role": "system", "content": system_prompt}]
            # Ajouter les derniers échanges pour le contexte
            for h in historique[-4:]:
                messages.append({"role": h["role"], "content": h["content"]})
            messages.append({"role": "user", "content": question})

            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=450,
                temperature=0.7,
            )
            texte = completion.choices[0].message.content
            if texte:
                return texte.strip()
        except Exception:
            # Fallback immédiat et transparent sur le moteur local
            pass

    # Utilisation du moteur local tunisien haute précision
    return repondre_en_tunisien_moteur_local(question, contexte)
