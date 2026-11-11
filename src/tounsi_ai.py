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
    "testville": "Tunis",
    "bizert": "Bizerte",
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
    "madhalla", "tsob", "tsab", "tsaksek", "chta", "chta 9wiya", "chta khfifa", "chta m3tadla",
    "chta barcha", "chta chwaya", "chta tji", "chta tebda", "chta te5fi", "chta twalli", "chettel",
    "bech techtel", "techtel", "chta lyoum", "chta ghodwa", "chta lbera7", "matar", "9atrat matar",
    "9atra", "9tar", "ch7al men chta", "rchach", "chta rchach", "chta t9ila", "chta 7amya",
    "sayl", "sayl kbir", "sayl 9wi", "tri9 ghar9a", "me barcha", "me tale3", "me tale3 fi tri9",
    "مطاف", "مطر", "نوة", "شتاء", "تصب"
]

MOTS_CLES_FROID = [
    "bard", "berd", "bareda", "bared", "zmigri", "garsa", "glacon", "froid",
    "barda", "barda 9wiya", "barda kbira", "barda sghira", "barda fi lil", "barda fi sbe7",
    "barda fi chta", "brouda", "berred", "berred kbir", "berred sghir", "برودة", "برد"
]

MOTS_CLES_CHAUD = [
    "s5ana", "skhana", "s5oun", "skhoun", "chams", "chems", "9ayla", "gayla", "chhar kawn",
    "chaouba", "chawba", "chaud", "chaleur", "skhoun", "7ami", "bared", "skhouna", "skhouna 9wiya",
    "chams 9wiya", "chams 7ar9a", "chams ghabt", "chams tale3et", "chams tla3", "chourou9 echams",
    "ghroub echams", "7ar9a", "7ar9a men chams", "darbet chams", "سخانة", "سخون", "شمس", "قائلة"
]

MOTS_CLES_VENT = [
    "rya7", "ryeh", "rih", "3ejja", "3ajja", "ghbar", "vent", "souffle",
    "ri7", "ri7 9wiya", "ri7 khfifa", "ri7 barda", "ri7 skhouna", "ri7 men el b7ar",
    "ri7 men el jbel", "ri7 men el janoub", "ri7 men ech chamal", "ri7 men ech char9",
    "ri7 men el gharb", "ri7 m3akra", "ri7 tay7a", "ri7 tale3a", "3asfa", "3asfa kbira",
    "3asfa 9wiya", "3asfa fi b7ar", "3asifa", "3asifa ra3diya", "3asifa ramliya", "3asifa taljiya",
    "ريح", "عجاج", "رياح"
]

MOTS_CLES_MER = [
    "b7ar", "bhar", "ba7r", "3wem", "3wema", "3oum", "mouj", "mer", "baignade", "plage",
    "b7ar hali", "b7ar hajj", "b7ar mawj", "mawj", "mawj kbir", "mawj 9wi", "mawj sghir",
    "med", "jzr", "9ar", "بحر", "عوم", "عومة", "موج"
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
    "bslama", "sbe7", "mse", "khir", "7al", "mrigel", "3lech", "wa9tech",
    "kifech", "win", "men", "3la", "fi", "m3a", "bla", "barcha", "chwaya", "yezzi", "tawa",
    "ghodwa", "lbera7", "lyoum", "bekri", "3chiyya", "lil", "nhar", "sma", "chams", "9mar",
    "nejma", "chta", "ri7", "ra3d", "bard", "skhoun", "7ami", "zine", "9bi7", "kbir", "sghir",
    "twil", "9sir", "9wi", "dhi3if", "jdid", "9dim", "mli7", "5ayeb", "bnin", "m7eb", "ghali",
    "rkhis", "sahla", "sa3ba", "wja3", "mrid", "7ay", "mayet", "dar", "bab", "chobbek", "tab",
    "korsi", "firch", "m5adda", "tay", "9ahwa", "me", "5obz", "kosksi", "3cha", "ftour", "8da",
    "l7em", "7out", "5dha", "fella", "3neb", "tffa7", "9ares", "7lib", "jben", "3sel", "mel7",
    "sokkar", "zit", "3ojja", "7rira", "brik", "mlewi", "msemen", "lablabi", "chakchouka",
    "keftaji", "ommi", "baba", "5ouya", "5ti", "3ayla", "sa7bi", "rajel", "mra", "wild", "bint",
    "tfol", "3juz", "5alti", "3ammi", "khal", "5ali", "jeddi", "jedda", "9alb", "rasi", "3in",
    "wjeh", "yed", "rjel", "kerch", "dher", "snan", "cha3r", "dm", "nfs", "3a9l", "7ob", "5ouf",
    "ferh", "7zen", "ghech", "9owwa", "7ila", "9dar", "9rib", "b3id", "fo9", "te7t", "9oddem",
    "wara", "9bal", "ba3d", "ymin", "sar", "wast", "jenb", "bin", "5edma", "9raya", "jem3a",
    "souk", "7anout", "maktab", "ma9ha", "7ammam", "tobbib", "9arya", "mdina", "tri9", "kareta",
    "karhba", "tayara", "babor", "matar", "mahatta", "b7ar", "jbel", "ghaba", "wad", "bir", "jnan",
    "chajra", "warga", "zr3a", "7jar", "rmel", "nar", "dwa", "dhil", "ena", "enti", "entouma",
    "houwa", "hiya", "houma", "7na", "mte3i", "mte3ek", "3andi", "3andek", "3andou", "3andha",
    "3andna", "3andkom", "3andhom", "hani", "haka", "hetha", "hethika",
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
    "fech ta3mel", "fesh ta3mel", "chera rak ta3mel", "ech ta3mel", "esh ta3mel",
    "9adech", "9addech", "9addch", "9deh", "9dhe", "9adeh 3omrek", "9addech 3omrek", "9adech 3mornek", "9addch 3omrek", "omrek", "age", "quel age", "quel âge", "age tu as", "tu as quel", "عمرك", "قداش عمرك",
    "شكون أنت", "من أنت", "عرف بنفسك", "فش تعمل", "إش تعمل"
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
    if "ghodwa" in t or "demain" in t or "غدوة" in t or "8odwa" in t or "8doua" in t:
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


def analyser_et_apprendre_inconnu(question: str) -> tuple[bool, str]:
    """
    Détecte et analyse les équivalences de type 'mot1=mot2' ou 'terme1=terme2'.
    Si détecté, injecte automatiquement l'association dans le dictionnaire ou code source.
    """
    q_str = question.strip()
    match = re.search(r"([a-zA-Zà-ûÀ-Û0-9_-]+)\s*=\s*([a-zA-Zà-ûÀ-Û0-9_-]+)", q_str)
    if match:
        terme_errone = match.group(1).lower().strip()
        terme_correct = match.group(2).lower().strip()

        # Mettre à jour MAPPING_VILLES_TUNISIENNES si c'est une ville ou équivalence
        if terme_correct in MAPPING_VILLES_TUNISIENNES.values() or terme_correct in MAPPING_VILLES_TUNISIENNES:
            val_canonique = MAPPING_VILLES_TUNISIENNES.get(terme_correct, terme_correct)
            MAPPING_VILLES_TUNISIENNES[terme_errone] = val_canonique

            # Injection physique dans le fichier source src/tounsi_ai.py pour persistance
            try:
                chemin_fichier = os.path.abspath(__file__)
                with open(chemin_fichier, "r", encoding="utf-8") as f:
                     contenu = f.read()

                # Vérifier si le terme n'existe pas déjà
                if f'"{terme_errone}":' not in contenu and f"'{terme_errone}':" not in contenu:
                    # Trouver la ligne de mapping des villes et ajouter le nouveau mapping
                    ancre = 'MAPPING_VILLES_TUNISIENNES = {'
                    if ancre in contenu:
                        contenu = contenu.replace(ancre, ancre + f'\n    "{terme_errone}": "{val_canonique}",')
                        with open(chemin_fichier, "w", encoding="utf-8") as f:
                            f.write(contenu)
            except Exception:
                pass

            return True, f"🚨 **Alerte Auto-Apprentissage / Correction IA** 🤖\n\nFhimtkom mli7 ! 3malt analyse lel message mte3ek w l9it elli **'{terme_errone}'** = **'{terme_correct}'**.\n\n✅ 9aydeha fel code source w f dictionnaire mte3i automatiquement ! Men tawa, kif nktb '{terme_errone}' n(w)elli n3rafha w nsalteha direct b s7i7 ! 🇹🇳"
        else:
            # Équivalence générale de mots
            MAPPING_VILLES_TUNISIENNES[terme_errone] = terme_correct
            return True, f"🚨 **Alerte Auto-Apprentissage** 🤖\n\nFhimt elli **'{terme_errone}'** t3ni **'{terme_correct}'**. 9aydeha f mémoire mte3i w 3malt mise à jour!"

    return False, ""

DICTIONNAIRE_TRADUCTION_DERJA = {
    "mtar": "pluie",
    "matar": "pluie",
    "chta": "pluie",
    "tsob": "pleuvoir",
    "tsab": "pleuvoir",
    "bard": "froid",
    "berd": "froid",
    "zmigri": "grand froid",
    "s5ana": "chaleur",
    "skhana": "chaleur",
    "s5oun": "chaud",
    "skhoun": "chaud",
    "chams": "soleil",
    "rya7": "vent",
    "rih": "vent",
    "b7ar": "mer",
    "bhar": "mer",
    "ghodwa": "demain",
    "8odwa": "demain",
    "8doua": "demain",
    "lyoum": "aujourd'hui",
    "ba3d ghodwa": "après-demain",
    "ta9s": "temps",
    "jaw": "temps",
    "w9tech": "quand",
    "kifech": "comment",
    "win": "où",
    "chnowa": "quel",
    "9adech": "combien"
}


def traduire_derja_vers_standard(texte: str) -> tuple[str, dict[str, str]]:
    """Traduit le texte en dialecte tunisien (Derja/Arabizi) vers un lexique standardisé pour l'IA."""
    t_clean = _nettoyer_texte(texte)
    mots = t_clean.split()
    traductions = {}
    texte_traduit = t_clean

    for mot in mots:
        if mot in DICTIONNAIRE_TRADUCTION_DERJA:
            trad = DICTIONNAIRE_TRADUCTION_DERJA[mot]
            traductions[mot] = trad
            texte_traduit = texte_traduit.replace(mot, trad)

    for phrase_derja, phrase_std in [
        ("ba3d ghodwa", "après-demain"),
        ("chta tsob", "il pleut"),
        ("ta9s s5oun", "temps chaud"),
        ("ta9s bard", "temps froid"),
        ("rih 9wi", "vent fort")
    ]:
        if phrase_derja in t_clean:
            texte_traduit = texte_traduit.replace(phrase_derja, phrase_std)
            traductions[phrase_derja] = phrase_std

    return texte_traduit, traductions


def deep_search_complet_mots(question: str) -> dict[str, Any]:
    """
    Effectue un Deep Search exhaustif sur TOUS les mots de la demande de l'utilisateur.
    Analyse chaque mot, calcule la similarité avec l'ensemble des dictionnaires (météo, villes, intentions)
    et extrait chaque entité pertinente pour une compréhension totale sans perte.
    """
    t_clean = _nettoyer_texte(question)
    mots = t_clean.split()

    resultats_mots = []
    entites_detectees = {
        "villes": [],
        "intentions": [],
        "traductions": {},
        "scores_mots": {}
    }

    # 1. Traduction globale
    _, traductions = traduire_derja_vers_standard(question)
    entites_detectees["traductions"] = traductions

    # 2. Analyse de chaque mot individuel (Deep Search exhaustif)
    tous_les_vocabulaires = {
        "pluie": MOTS_CLES_PLUIE,
        "froid": MOTS_CLES_FROID,
        "chaud": MOTS_CLES_CHAUD,
        "vent": MOTS_CLES_VENT,
        "mer": MOTS_CLES_MER,
        "vetements": MOTS_CLES_VETEMENTS,
        "sortie": MOTS_CLES_SORTIE,
        "salutation": MOTS_CLES_SALUTATION
    }

    for mot in mots:
        meilleur_score_mot = 0.0
        categorie_trouvee = "inconnu"

        for cat, liste_mots in tous_les_vocabulaires.items():
            for ref in liste_mots:
                sim = similarite_caracteres(mot, ref)
                if sim > meilleur_score_mot:
                    meilleur_score_mot = sim
                    categorie_trouvee = cat

        entites_detectees["scores_mots"][mot] = {
            "categorie": categorie_trouvee,
            "score": round(meilleur_score_mot, 2)
        }

        if meilleur_score_mot >= 0.5 and categorie_trouvee not in entites_detectees["intentions"]:
            entites_detectees["intentions"].append(categorie_trouvee)

    # 3. Détection de ville par deep search sur chaque mot
    for mot in mots:
        for v_cle, v_nom in MAPPING_VILLES_TUNISIENNES.items():
            if similarite_caracteres(mot, v_cle) >= 0.8 or mot == v_cle:
                if v_nom not in entites_detectees["villes"]:
                    entites_detectees["villes"].append(v_nom)

    return {
        "mots_analyses": mots,
        "total_mots": len(mots),
        "entites": entites_detectees,
        "analyse_exhaustive": f"DeepSearch: {len(mots)} mots passés au peigne fin. Intentions: {entites_detectees['intentions']}, Villes: {entites_detectees['villes']}."
    }
def distance_levenshtein(s1: str, s2: str) -> int:
    """Calcule la distance de Levenshtein entre deux mots."""
    if len(s1) < len(s2):
        return distance_levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def similarite_caracteres(s1: str, s2: str) -> float:
    """Calcule un score de similarité par n-grammes de caractères et sous-séquences (0.0 à 1.0)."""
    if not s1 or not s2:
        return 0.0
    # Jaccard sur les bigrammes de caractères pour chaque mot
    set1 = {s1[i:i+2] for i in range(len(s1)-1)} if len(s1) >= 2 else {s1}
    set2 = {s2[i:i+2] for i in range(len(s2)-1)} if len(s2) >= 2 else {s2}
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    jaccard = intersection / union if union > 0 else 0.0

    # Levenshtein normalisé
    dist = distance_levenshtein(s1, s2)
    max_len = max(len(s1), len(s2))
    lev_score = 1.0 - (dist / max_len) if max_len > 0 else 1.0

    return 0.5 * jaccard + 0.5 * lev_score


def contient_mot(texte: str, mots: list[str], max_dist: int = 2, seuil_sim: float = 0.6) -> bool:
    """
    Analyse approfondie :
    1. Correspondance exacte.
    2. Comparaison caractère par caractère (Levenshtein + bigrammes) sur chaque mot du texte.
    3. Détection de sous-chaînes imbriquées (lettre par lettre).
    """
    mots_texte = texte.split()
    for m in mots:
        pattern = r"(?<!\w)" + re.escape(m) + r"(?!\w)"
        if re.search(pattern, texte) or m in texte:
            return True
        for mot_t in mots_texte:
            if abs(len(mot_t) - len(m)) <= max_dist:
                if distance_levenshtein(mot_t, m) <= max_dist:
                    return True
                if similarite_caracteres(mot_t, m) >= seuil_sim:
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

    # État / Prendre des nouvelles ("chna7welek", "cv", "labes") - MAIS PAS si la phrase contient une demande météo sur un gouvernorat (ex: "chn7wl ta9s f nabeul")
    indices_meteo = (
        MOTS_CLES_PLUIE + MOTS_CLES_FROID + MOTS_CLES_CHAUD + MOTS_CLES_VENT +
        MOTS_CLES_MER + MOTS_CLES_VETEMENTS + MOTS_CLES_SORTIE + [
            "ta9s", "taqs", "te9s", "meteo", "météo", "jaw", "jawha", "climat", "7arara", "harara",
            "darajat", "temperature", "température", "degre", "degré", "ghodwa", "demain",
            "lyoum", "ba3d ghodwa", "weekend", "chams", "s7ab", "chn7wl", "chnwa", "shwaya", "طقس", "جو", "حرارة", "غدوة", "مطر"
        ]
    )

    if contient_mot(texte_clean, MOTS_CLES_ETAT) and not (detecter_gouvernorat(texte_clean) or contient_mot(texte_clean, indices_meteo)):
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
    # Un message court avec juste un nom de ville (ex: "nabeul", "o f nabeul ?") n'est PAS conversationnel
    if len(texte_clean.split()) <= 3 and detecter_gouvernorat(texte_clean):
        return False, ""

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

def deep_think_analyser_demande(question: str, intentions: list[str]) -> dict:
    """Analyse approfondie multi-hypothèses de la demande météo."""
    return {
        "intentions_detectees": intentions,
        "complexite": "elevee" if len(intentions) > 2 else "normale",
        "analyse_profonde": True,
        "mots_cles_trouves": len(intentions)
    }


def repondre_en_tunisien_moteur_local(question: str, contexte: dict[str, Any]) -> str:
    """Génère une réponse complète, fluide et authentique en derja tunisienne."""
    q_clean = _nettoyer_texte(question)

    # Traduction automatique du Derja / Arabizi avant analyse
    texte_traduit, lexique_traduit = traduire_derja_vers_standard(q_clean)
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
            if "omr" in q_clean or "3omr" in q_clean or "age" in q_clean or "9adech" in q_clean or "9addech" in q_clean:
                return f"Ena model AI makhdoum jdid, ma 3andiuch 3omr kima el bchar, ama nectel 24/24 w dima jdid 3al la7dha ! 🇹🇳"
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

    # Détection de l'intention de la question avec Deep Think
    intentions_possibles = []
    if detecter_question_geographie(question): intentions_possibles.append("geographie")
    if contient_mot(q_clean, MOTS_CLES_PLUIE): intentions_possibles.append("pluie")
    if contient_mot(q_clean, MOTS_CLES_FROID): intentions_possibles.append("froid")
    if contient_mot(q_clean, MOTS_CLES_CHAUD): intentions_possibles.append("chaud")
    if contient_mot(q_clean, MOTS_CLES_VENT): intentions_possibles.append("vent")
    if contient_mot(q_clean, MOTS_CLES_MER): intentions_possibles.append("mer")
    if contient_mot(q_clean, MOTS_CLES_VETEMENTS): intentions_possibles.append("vetements")
    if contient_mot(q_clean, MOTS_CLES_SORTIE): intentions_possibles.append("sortie")
    if contient_mot(q_clean, MOTS_CLES_SALUTATION): intentions_possibles.append("salutation")

    deep_result = deep_think_analyser_demande(question, intentions_possibles if intentions_possibles else ["general"])

    a_geographie = "geographie" in intentions_possibles
    a_pluie = "pluie" in intentions_possibles
    a_froid = "froid" in intentions_possibles
    a_chaud = "chaud" in intentions_possibles
    a_vent = "vent" in intentions_possibles
    a_mer = "mer" in intentions_possibles
    a_vetements = "vetements" in intentions_possibles
    a_sortie = "sortie" in intentions_possibles
    a_salut = "salutation" in intentions_possibles

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
    mode_ia: str = "Local Derja (Défaut)",
    ollama_url: str = "http://localhost:11434",
    ollama_model: str = "deepseek-r1:7b",
) -> str:
    """
    Point d'entrée principal :
    1. Vérifie si le message contient une règle d'équivalence ('mot1=mot2') pour l'auto-apprentissage direct.
    2. Vérifie si le message est conversationnel (salutations, identité, remerciements, etc.)
    3. Détecte le gouvernorat mentionné (ou utilise le gouvernorat actif)
    4. Récupère les vraies données météo
    5. Route vers Ollama (DeepSeek-R1), OpenAI ou moteur local selon le choix utilisateur.
    """
    # 0. Vérification règle d'auto-apprentissage explicite ('mot1=mot2') ou signalement "erreur"
    appris, msg_apprentissage = analyser_et_apprendre_inconnu(question)
    if appris:
        return msg_apprentissage

    q_clean = _nettoyer_texte(question)
    if q_clean == "erreur" and historique:
        dernier_user = next((h["content"] for h in reversed(historique) if h["role"] == "user" and h["content"].lower() != "erreur"), "")
        contexte_corrige = recuperer_contexte_meteo(detecter_gouvernorat(dernier_user, gouvernorat_par_defaut=gouvernorat_actif))
        reponse_corrigee = repondre_en_tunisien_moteur_local(dernier_user, contexte_corrige)
        return (
            f"🔄 **Auto-Réévaluation & Correction Automatique** 🤖\n\n"
            f"Fhimt elli el jeweb el 9dim ken ghalat (3la demande: *'{dernier_user}'*).\n"
            f"3malt comparaison w ré-analyse lel manti9, w hathi r-réponse s7iha el jdida :\n\n"
            f"---\n{reponse_corrigee}\n---\n\n"
            f"✅ Kif t3addi tawa message nromal, nzid nethaqqaq w nstahfedh bel iṣla7!"
        )
    ville_cible = detecter_gouvernorat(question, gouvernorat_par_defaut=gouvernorat_actif)

    # Suppression du retour anticipé 'est_conv' pour laisser Ollama traiter toutes les requêtes si sélectionné
    est_conv, _ = est_message_purement_conversationnel(q_clean)
    contexte = recuperer_contexte_meteo(ville_cible)
    if est_conv and "Ollama" not in mode_ia and "OpenAI" not in mode_ia:
        return repondre_en_tunisien_moteur_local(question, contexte)

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
Tu es un assistant IA météorologique tunisien authentique nommé "Tounsi Météo AI".
Tu t'exprimes EXCLUSIVEMENT en dialecte tunisien (Derja en alphabet latin / Arabizi naturel et expressif, ex: "Ahla bik ya sidi", "El ta9s lyoum s5on w fama ryah", "Lézem t هز madhalla").
Interdiction absolue de répondre en français littéraire ou en arabe classique : utilise uniquement le vrai parler tunisien de tous les jours.

Données géographiques et météo réelles pour {ville_cible} :
{resume_meteo}

Règles strictes :
1. Parle 100% en Derja tunisienne authentique et chaleureuse.
2. Utilise les données météo et géographiques ci-dessus pour répondre précisément aux questions sur le temps, les températures, la pluie, le vent ou la position géographique ("win jéya").
3. Donne des conseils pratiques adaptés (vêtements, sortie, baignade, parapluie).
4. Sois concis, direct et utilise des emojis météo appropriés (☀️, 🌧️, 💨, 🌊, 🌡️).
"""

    # Appel exclusif à Ollama (DeepSeek-R1) avec timeout élargi et tuning
    try:
        import requests
        prompt_complet = f"System:\n{system_prompt}\n\n"
        for h in historique[-4:]:
            prompt_complet += f"{h['role'].capitalize()}: {h['content']}\n"
        prompt_complet += f"User: {question}\nAssistant:"

        payload = {
            "model": ollama_model,
            "prompt": prompt_complet,
            "stream": False,
            "options": {
                "temperature": 0.6,
                "top_p": 0.9,
                "num_predict": 500,
            }
        }
        resp = requests.post(f"{ollama_url.rstrip('/')}/api/generate", json=payload, timeout=120)
        if resp.status_code == 200:
            data = resp.json()
            reponse_brute = data.get("response", "")
            reponse_propre = re.sub(r'<think>.*?</think>', '', reponse_brute, flags=re.DOTALL).strip()
            if reponse_propre:
                return reponse_propre
    except Exception as e:
        return f"⚠️ Erreur de connexion avec Ollama ({ollama_url}) : {e}"
