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

import unicodedata
from typing import Any

# --------------------------------------------------------------------------
# Données géographiques complètes des 24 gouvernorats de la Tunisie
# --------------------------------------------------------------------------
INFOS_GEOGRAPHIQUES: dict[str, dict[str, Any]] = {
    "Ariana": {
        "lat": 36.8665,
        "lon": 10.1647,
        "nom_ar": "أريانة",
        "region": "Grand Tunis",
        "zone": "Nord",
        "milieu": "Plaine littorale et collines périurbaines",
        "altitude_m": 15,
        "facade_maritime": True,
        "voisins": ["Tunis", "Manouba", "Bizerte"],
        "description_fr": "Située dans la banlieue nord de Tunis, s'étendant des collines d'Ennasr jusqu'aux plages de Raoued et Kalâat el-Andalous le long de la basse vallée de la Medjerda.",
        "description_derja": "Chamel el 3assima (Grand Tunis), ba7dha Tunis w Manouba w toftor 3al b7ar fi Raoued w Kalâat el-Andalous.",
        "climat": "Méditerranéen côtier doux et tempéré, brise marine et plaines agricoles fertiles.",
    },
    "Béja": {
        "lat": 36.7256,
        "lon": 9.1817,
        "nom_ar": "باجة",
        "region": "Nord-Ouest",
        "zone": "Nord",
        "milieu": "Plaines céréalières et collines du Tell (grenier à blé)",
        "altitude_m": 220,
        "facade_maritime": True,
        "voisins": ["Bizerte", "Manouba", "Zaghouan", "Siliana", "Jendouba"],
        "description_fr": "Cœur céréalier de la Tunisie (grenier de Rome), traversée par l'oued Medjerda, s'étendant des riches collines intérieures jusqu'à la côte nord méditerranéenne (Nefza, Cap Négro).",
        "description_derja": "Chamel gharbi (el Tell), ma3roufa b fela7et el 9am7 (sabit tounes), feydha Medjerda, bin Bizerte, Jendouba w Siliana w toftor 3al b7ar f Nefza.",
        "climat": "Subhumide, pluies abondantes en hiver et printemps, terres très fertiles et vallons verdoyants.",
    },
    "Ben Arous": {
        "lat": 36.7531,
        "lon": 10.2189,
        "nom_ar": "بن عروس",
        "region": "Grand Tunis",
        "zone": "Nord",
        "milieu": "Plaine littorale et contreforts montagneux (Boukornine)",
        "altitude_m": 25,
        "facade_maritime": True,
        "voisins": ["Tunis", "Manouba", "Zaghouan", "Nabeul"],
        "description_fr": "Au sud du Grand Tunis, bordée par la baie de Tunis (Radès, Hammam Lif, Ezzahra) et adossée au massif remarquable du Djebel Boukornine.",
        "description_derja": "Janoub el 3assima (Grand Tunis), feha b7ar (Radès, Ezzahra, Hammam Lif) w jbal Boukornine, bin Tunis, Zaghouan w Nabeul.",
        "climat": "Méditerranéen maritime, influences combinées de la mer et du relief forestier de Boukornine.",
    },
    "Bizerte": {
        "lat": 37.2744,
        "lon": 9.8739,
        "nom_ar": "بنزرت",
        "region": "Nord",
        "zone": "Nord",
        "milieu": "Littoral maritime septentrional et lagunes (Ichkeul)",
        "altitude_m": 5,
        "facade_maritime": True,
        "voisins": ["Ariana", "Manouba", "Béja"],
        "description_fr": "Pointe la plus septentrionale de toute l'Afrique (Ras Angela), bordée par une côte rocheuse sauvage, le lac de Bizerte et le parc national d'Ichkeul.",
        "description_derja": "A9sa chamel tounes w ifri9ya kolha (Ras Angela), 3al b7ar l'abiyadh l'moutawasset, feha bo7ayret Ichkeul w Benzart, ba7dha Béja w Ariana.",
        "climat": "Méditerranéen maritime humide et très venté, hivers frais et arrosés, étés doux aérés par le vent du large.",
    },
    "Gabès": {
        "lat": 33.8815,
        "lon": 10.0982,
        "nom_ar": "قابس",
        "region": "Sud-Est",
        "zone": "Sud",
        "milieu": "Oasis côtière maritime unique en Méditerranée",
        "altitude_m": 10,
        "facade_maritime": True,
        "voisins": ["Sfax", "Sidi Bouzid", "Gafsa", "Kébili", "Médenine"],
        "description_fr": "Au fond du golfe de Gabès, abrite l'unique oasis maritime de la Méditerranée (Chenini) où la palmeraie côtoie directement la mer avec les plus fortes marées de Tunisie.",
        "description_derja": "Janoub char9i 3al b7ar (khalij Gabès), el waha el ba7riya el wa7ida fel 3alem el motawasset li ba7dha el ba7r toul, bin Sfax w Médenine.",
        "climat": "Aride adouci par la mer, étés chauds mais tempérés par la brise marine du golfe, hivers doux.",
    },
    "Gafsa": {
        "lat": 34.4250,
        "lon": 8.7842,
        "nom_ar": "قفصة",
        "region": "Sud-Ouest",
        "zone": "Sud",
        "milieu": "Bassin pré-saharien montagnard et oasis",
        "altitude_m": 298,
        "facade_maritime": False,
        "voisins": ["Kasserine", "Sidi Bouzid", "Gabès", "Kébili", "Tozeur"],
        "description_fr": "Carrefour stratégique du Sud-Ouest entre la steppe centrale et le Sahara, entourée des chaînes rocheuses du Djebel Orbata et Djebel Bou Ramli, riche en bassins miniers et oasis.",
        "description_derja": "Janoub gharbi, bab el sa7ra w fassil ma bin el wasat wel janoub, feha wé7at w jbel el phosphate, bin Kasserine, Tozeur w Kebili.",
        "climat": "Aride à semi-aride continental, amplitude thermique marquée entre le jour et la nuit, étés très chauds.",
    },
    "Jendouba": {
        "lat": 36.5011,
        "lon": 8.7802,
        "nom_ar": "جندوبة",
        "region": "Nord-Ouest",
        "zone": "Nord",
        "milieu": "Massifs montagneux de Kroumirie et côte de corail",
        "altitude_m": 140,
        "facade_maritime": True,
        "voisins": ["Béja", "Le Kef"],
        "description_fr": "À l'extrême Nord-Ouest le long de la frontière algérienne, couverte des forêts denses de chênes-lièges de Kroumirie (Aïn Draham) et du littoral de Tabarka.",
        "description_derja": "A9sa chamel gharbi 3al 7oudoud el jazayriya, fiha jbel Kroumirie, Aïn Draham w b7ar Tabarka, a3la blasa tssob feha el mtar wel thalj f tounes.",
        "climat": "Région la plus humide de Tunisie (jusqu'à 1500 mm de pluie/an), hivers froids avec chutes régulières de neige à Aïn Draham.",
    },
    "Kairouan": {
        "lat": 35.6781,
        "lon": 10.0963,
        "nom_ar": "القيروان",
        "region": "Centre",
        "zone": "Centre",
        "milieu": "Vaste plaine steppique intérieure",
        "altitude_m": 68,
        "facade_maritime": False,
        "voisins": ["Zaghouan", "Siliana", "Sousse", "Mahdia", "Sfax", "Sidi Bouzid"],
        "description_fr": "Capitale spirituelle située dans la grande plaine centrale entre les bassins des oueds Merguellil et Zeroud, sans accès à la mer.",
        "description_derja": "Wasat el dakhili (el Steppes), shta7 wase3 b3ida 3al b7ar, bin Soussa, Siliana w Sidi Bouzid, ta9sha cheye7 w s5oun fel sif.",
        "climat": "Semi-aride chaud continental, étés très chauds sous influence fréquente du sirocco (chehili), faible pluviométrie.",
    },
    "Kasserine": {
        "lat": 35.1676,
        "lon": 8.8365,
        "nom_ar": "القصرين",
        "region": "Centre-Ouest",
        "zone": "Centre",
        "milieu": "Hautes steppes occidentales et point culminant (Chambi)",
        "altitude_m": 650,
        "facade_maritime": False,
        "voisins": ["Le Kef", "Siliana", "Sidi Bouzid", "Gafsa"],
        "description_fr": "Frontalière de l'Algérie, abrite le toit de la Tunisie : le Djebel Chambi culminant à 1 544 m au milieu des steppes d'alfa et des forêts de pins d'Alep.",
        "description_derja": "Wasat gharbi 3al 7oudoud m3a el dzayer, feha a3la 9emma f tounes (Jbel Chambi 1544m), steppes 3alya, bard kbir fel chta w s5ana chey7a fel sif.",
        "climat": "Semi-aride froid d'altitude, hivers rigoureux et froids avec gelées et neige occasionnelle, étés secs et chauds.",
    },
    "Kébili": {
        "lat": 33.7044,
        "lon": 8.9690,
        "nom_ar": "قبلي",
        "region": "Sud-Ouest",
        "zone": "Sud",
        "milieu": "Région du Nefzaoua, oasis et Chott el-Jérid",
        "altitude_m": 45,
        "facade_maritime": False,
        "voisins": ["Tozeur", "Gafsa", "Gabès", "Médenine", "Tataouine"],
        "description_fr": "Aux portes du Grand Erg Oriental et bordée par l'immensité saline du Chott el-Jérid, abrite de vastes palmeraies productrices de dattes Deglet Nour.",
        "description_derja": "Janoub tounsi (Nefzaoua), dakhla fel sa7ra w Chott el Jérid, a9reb blasa lil 3erg el kbir, ta9s sa7rawi 100% cheye7 w s5oun.",
        "climat": "Saharien désertique chaud, précipitations très rares, étés torrides dépassant 45°C, nuits fraîches en hiver.",
    },
    "Le Kef": {
        "lat": 36.1822,
        "lon": 8.7149,
        "nom_ar": "الكاف",
        "region": "Nord-Ouest",
        "zone": "Nord",
        "milieu": "Haut Tell montagneux et plateaux calcaires",
        "altitude_m": 620,
        "facade_maritime": False,
        "voisins": ["Jendouba", "Siliana", "Kasserine"],
        "description_fr": "Bâtie en nid d'aigle sur le flanc sud du Djebel Dyr à plus de 600 m d'altitude près de la frontière algérienne, célèbre pour ses tables calcaires (Table de Jugurtha).",
        "description_derja": "Chamel gharbi dakhili (el Haut Tell), manti9a jabaliya 3alya 3al 7doud m3a el dzayer (Jougourtha), ta9sha continental bared fel chta.",
        "climat": "Méditerranéen continental d'altitude, hivers froids avec gelées fréquentes et neige, étés chauds et aérés.",
    },
    "Mahdia": {
        "lat": 35.5047,
        "lon": 11.0622,
        "nom_ar": "المهدية",
        "region": "Centre-Est",
        "zone": "Centre",
        "milieu": "Presqu'île maritime du Sahel méridional",
        "altitude_m": 8,
        "facade_maritime": True,
        "voisins": ["Sousse", "Monastir", "Sfax", "Kairouan"],
        "description_fr": "Ancienne capitale fatimide sur une presqu'île rocheuse (Cap Afrique) s'avançant dans la mer Méditerranée, bordée par des plages de sable fin et des oliveraies intérieures.",
        "description_derja": "Janoub el Sahel el char9i, 3al b7ar (Cap Afrique), ma bin Monastir w Sfax, ma3roufa b sayd el 7out wel zaytoun.",
        "climat": "Méditerranéen maritime doux, hivers tempérés par la mer, étés agréables grâce à la brise marine constante.",
    },
    "Manouba": {
        "lat": 36.8080,
        "lon": 10.0972,
        "nom_ar": "منوبة",
        "region": "Grand Tunis",
        "zone": "Nord",
        "milieu": "Basse vallée de la Medjerda et plaine agricole",
        "altitude_m": 50,
        "facade_maritime": False,
        "voisins": ["Tunis", "Ariana", "Ben Arous", "Zaghouan", "Béja", "Bizerte"],
        "description_fr": "À l'ouest immédiat de Tunis, plaine fertile maraîchère et arboricole irriguée par le cours inférieur de l'oued Medjerda.",
        "description_derja": "Gharb el 3assima (Grand Tunis), manti9a fela7iya mte3 sehel Medjerda, bin Tunis, Béja w Bizerte.",
        "climat": "Méditerranéen franc, hivers doux et étés chauds avec pluviométrie modérée.",
    },
    "Médenine": {
        "lat": 33.3549,
        "lon": 10.5055,
        "nom_ar": "مدنين",
        "region": "Sud-Est",
        "zone": "Sud",
        "milieu": "Plaine littorale de la Jeffara, île de Djerba et monts des Ksour",
        "altitude_m": 90,
        "facade_maritime": True,
        "voisins": ["Gabès", "Kébili", "Tataouine"],
        "description_fr": "Englobe la péninsule de Zarzis, la plaine de la Jeffara et la célèbre île de Djerba face au golfe de Gabès, s'adossant aux ksour berbères.",
        "description_derja": "Janoub char9i, feha jazirat Djerba w Zarzis wel Jeffara, m9abla khalij Gabès w 9riba l lybia, bin Gabès w Tataouine.",
        "climat": "Aride maritime très ensoleillé, étés chauds tempérés sur le littoral insulaire de Djerba, hivers doux.",
    },
    "Monastir": {
        "lat": 35.7780,
        "lon": 10.8262,
        "nom_ar": "المنستير",
        "region": "Centre-Est",
        "zone": "Centre",
        "milieu": "Presqu'île côtière du Sahel et sebkhas",
        "altitude_m": 4,
        "facade_maritime": True,
        "voisins": ["Sousse", "Mahdia"],
        "description_fr": "Presqu'île avancée sur le golfe de Hammamet bordée de falaises et de salines côtières, au cœur de la conurbation touristique du Sahel.",
        "description_derja": "Sahel char9i, chibh jazira 3al b7ar (khalij Monastir), ba7dha Sousse w Mahdia, jawha ba7ri mou3tadel.",
        "climat": "Méditerranéen maritime doux, faible amplitude thermique grâce à la mer, brises rafraîchissantes en été.",
    },
    "Nabeul": {
        "lat": 36.4561,
        "lon": 10.7376,
        "nom_ar": "نابل",
        "region": "Nord-Est",
        "zone": "Nord",
        "milieu": "Péninsule maritime du Cap Bon (verger de la Tunisie)",
        "altitude_m": 12,
        "facade_maritime": True,
        "voisins": ["Ben Arous", "Zaghouan", "Sousse"],
        "description_fr": "Occupe toute la péninsule fertile du Cap Bon entre le golfe de Tunis et le golfe de Hammamet, réputée pour ses agrumes, vignobles et longues côtes sableuses (Hammamet, Kélibia).",
        "description_derja": "Chamel char9i (chibh jazirat el Cap Bon), da5la fel b7ar ma bin khalij tounes w khalij hammamet, ma3roufa b Hammamet w Kelibia.",
        "climat": "Méditerranéen maritime doux et humide, hivers tempérés, précipitations régulières, étés chauds adoucis par les brises côtières.",
    },
    "Sfax": {
        "lat": 34.7406,
        "lon": 10.7603,
        "nom_ar": "صفاقس",
        "region": "Centre-Est",
        "zone": "Centre",
        "milieu": "Plaine littorale côtière et oliveraies (archipel de Kerkennah)",
        "altitude_m": 18,
        "facade_maritime": True,
        "voisins": ["Mahdia", "Kairouan", "Sidi Bouzid", "Gabès"],
        "description_fr": "Deuxième métropole économique du pays, immense façade maritime face aux îles Kerkennah sur le golfe de Gabès, célèbre pour ses millions d'oliviers (Sialine).",
        "description_derja": "Wasat char9i 3al khalij mte3 Gabès, m9abla jzayer Kerkennah, akbar ghaba mte3 zaytoun, bin Mahdia, Sidi Bouzid w Gabès.",
        "climat": "Semi-aride maritime, étés chauds modérés par la brise marine, ensoleillement record supérieur à 3 200 heures/an.",
    },
    "Sidi Bouzid": {
        "lat": 35.0382,
        "lon": 9.4849,
        "nom_ar": "سيدي بوزيد",
        "region": "Centre",
        "zone": "Centre",
        "milieu": "Basse steppe centrale et plaines agricoles irriguées",
        "altitude_m": 320,
        "facade_maritime": False,
        "voisins": ["Kairouan", "Siliana", "Kasserine", "Gafsa", "Sfax", "Gabès"],
        "description_fr": "Au cœur des vastes steppes centrales, haut lieu de maraîchage et d'arboriculture irriguée, entourée de djebels isolés (Kbar, Gouleb).",
        "description_derja": "Ghalb el wasat el tounsi (Basse steppe), fela7a kbiira mte3 5odhra w ghalla, bin Kairouan, Kasserine, Sfax w Gafsa.",
        "climat": "Semi-aride continental, étés chauds et ensoleillés, amplitudes thermiques marquées, faibles pluies.",
    },
    "Siliana": {
        "lat": 36.0850,
        "lon": 9.3708,
        "nom_ar": "سليانة",
        "region": "Nord-Ouest",
        "zone": "Nord",
        "milieu": "Montagnes du Tell et massifs forestiers (Bargou, Serj)",
        "altitude_m": 430,
        "facade_maritime": False,
        "voisins": ["Béja", "Jendouba", "Le Kef", "Kairouan", "Kasserine", "Sidi Bouzid", "Zaghouan"],
        "description_fr": "Zone de transition montagneuse entre le Tell et la haute steppe, abritant les parcs nationaux des Djebels Bargou et Serj ainsi que le barrage de Siliana.",
        "description_derja": "Chamel gharbi wasat, fiha jbel Bargou w jbel Serj, 9riba l Zaghouan, Béja, Kef w Kairouan, blasa feha fela7a w jbel.",
        "climat": "Semi-aride supérieur à subhumide selon l'altitude, hivers frais et humides, étés modérément chauds.",
    },
    "Sousse": {
        "lat": 35.8256,
        "lon": 10.6370,
        "nom_ar": "سوسة",
        "region": "Centre-Est",
        "zone": "Centre",
        "milieu": "Littoral maritime du Sahel (Perle du Sahel)",
        "altitude_m": 12,
        "facade_maritime": True,
        "voisins": ["Monastir", "Mahdia", "Kairouan", "Zaghouan", "Nabeul"],
        "description_fr": "« Perle du Sahel », bordant le golfe d'Hammamet avec de magnifiques plages de sable et un arrière-pays d'oliveraies séculaires.",
        "description_derja": "Wasat char9i (Jawharat el Sahel), 3al b7ar mte3 khalij Hammamet, ma3roufa b chwate2ha w zaytounha, bin Monastir, Mahdia, Kairouan w Nabeul.",
        "climat": "Méditerranéen maritime tempéré, étés chauds et ensoleillés régulés par la brise marine, hivers doux et agréables.",
    },
    "Tataouine": {
        "lat": 32.9297,
        "lon": 10.4518,
        "nom_ar": "تطاوين",
        "region": "Sud",
        "zone": "Sud",
        "milieu": "Monts des Ksour et grand désert saharien (Grand Erg)",
        "altitude_m": 250,
        "facade_maritime": False,
        "voisins": ["Kébili", "Médenine"],
        "description_fr": "Plus vaste gouvernorat de Tunisie (25 % de la superficie nationale), s'étendant des monts des Ksour troglodytiques jusqu'aux dunes du Grand Erg Oriental et Borj el Khadra aux confins sahariens.",
        "description_derja": "A9sa el janoub el sa7rawi (akbar gouvernorat b el misa7a), fiha jbel el Ksour w ghabet el rmal mte3 el sa7ra el kobra l 7ad Borj el Khadra.",
        "climat": "Hyper-aride désertique saharien, chaleur extrême l'été (> 45°C), vents de sable (sirocco) et nuits froides en hiver.",
    },
    "Tozeur": {
        "lat": 33.9197,
        "lon": 8.1335,
        "nom_ar": "توزر",
        "region": "Sud-Ouest",
        "zone": "Sud",
        "milieu": "Région du Jérid, grandes palmeraies et chotts salés",
        "altitude_m": 43,
        "facade_maritime": False,
        "voisins": ["Gafsa", "Kébili"],
        "description_fr": "Au cœur du Jérid entre le Chott el-Jérid et le Chott el-Gharsa, oasis saharienne millénaire célèbre pour ses palmeraies irriguées et son architecture en briques d'argile.",
        "description_derja": "Janoub gharbi (bled el Jérid), ba7dha Chott el Jérid w 3al 7doud m3a el dzayer, ma3roufa b wé7at el nakhil (deglet nour) w ta9s sa7rawi s5oun.",
        "climat": "Saharien aride chaud, ensoleillement exceptionnel, étés torrides et hivers doux le jour mais froids la nuit.",
    },
    "Tunis": {
        "lat": 36.8065,
        "lon": 10.1815,
        "nom_ar": "تونس",
        "region": "Grand Tunis",
        "zone": "Nord",
        "milieu": "Plaine littorale et collines côtières (golfe de Tunis)",
        "altitude_m": 10,
        "facade_maritime": True,
        "voisins": ["Ariana", "Ben Arous", "Manouba"],
        "description_fr": "Capitale de la Tunisie, située au fond du golfe de Tunis autour du lac de Tunis et de la sebkha Séjoumi, adossée aux collines du Belvédère et de Sidi Bou Saïd.",
        "description_derja": "Fi chamel char9i (Grand Tunis), 3al ba7r mte3 khalij tounes, daire biha Ariana, Manouba w Ben Arous.",
        "climat": "Méditerranéen maritime franc, hivers doux et arrosés, étés chauds et ensoleillés atténués par la brise marine.",
    },
    "Zaghouan": {
        "lat": 36.4029,
        "lon": 10.1429,
        "nom_ar": "زغوان",
        "region": "Nord-Est",
        "zone": "Nord",
        "milieu": "Montagnes de la Dorsale tunisienne (Djebel Zaghouan)",
        "altitude_m": 220,
        "facade_maritime": False,
        "voisins": ["Tunis", "Ben Arous", "Manouba", "Nabeul", "Siliana", "Kairouan", "Sousse"],
        "description_fr": "Au pied du Djebel Zaghouan (1 292 m), château d'eau historique de la Tunisie antique d'où partait l'aqueduc de Carthage, riche en sources thermales et parcs naturels.",
        "description_derja": "Chamel char9i dakhili, ma3roufa b Jbel Zaghouan (1292m) w 3youn el me (Temple des Eaux), bin Grand Tunis, Nabeul, Siliana w Soussa.",
        "climat": "Semi-aride à subhumide de montagne, sources abondantes, hivers frais et étés tempérés par le relief.",
    },
}

# 24 Gouvernorats de la Tunisie : nom -> (latitude, longitude)
VILLES: dict[str, tuple[float, float]] = {
    nom: (info["lat"], info["lon"]) for nom, info in INFOS_GEOGRAPHIQUES.items()
}

# Regroupement des gouvernorats par grande région
REGIONS_TUNISIE: dict[str, list[str]] = {
    "Grand Tunis": ["Tunis", "Ariana", "Ben Arous", "Manouba"],
    "Nord-Est": ["Nabeul", "Zaghouan", "Bizerte"],
    "Nord-Ouest": ["Béja", "Jendouba", "Le Kef", "Siliana"],
    "Centre-Est (Sahel)": ["Sousse", "Monastir", "Mahdia", "Sfax"],
    "Centre-Ouest": ["Kairouan", "Kasserine", "Sidi Bouzid"],
    "Sud-Ouest": ["Gafsa", "Tozeur", "Kébili"],
    "Sud-Est": ["Gabès", "Médenine", "Tataouine"],
}


def obtenir_info_geographique(nom: str) -> dict[str, Any]:
    """Retourne la fiche géographique complète d'un gouvernorat ou d'un point GPS."""
    ville_norm = normaliser_ville(nom)
    if ville_norm in INFOS_GEOGRAPHIQUES:
        return INFOS_GEOGRAPHIQUES[ville_norm]

    # Cas point GPS personnalisé (ex: GPS_36.80_10.18)
    lat, lon = VILLES.get(ville_norm, (36.8065, 10.1815))
    return {
        "lat": lat,
        "lon": lon,
        "nom_ar": ville_norm,
        "region": "Point géographique personnalisé",
        "zone": "Tunisie",
        "milieu": "Coordonnées GPS directes",
        "altitude_m": 50,
        "facade_maritime": False,
        "voisins": [],
        "description_fr": f"Point géographique situé aux coordonnées {lat:.4f}°N, {lon:.4f}°E en Tunisie.",
        "description_derja": f"Blasa b les coordonnées GPS {lat:.2f}°N, {lon:.2f}°E f tounes.",
        "climat": "Climat méditerranéen tunisien selon l'emplacement géographique.",
    }


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
