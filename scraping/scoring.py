"""
Scoring heuristique des annonces immobilieres — sans IA.

Produit les memes champs qu'enrichissement.py (enrichir_annonce_claude) :
  - etage, parking, balcon, vue_mer  (extraction par mots-cles)
  - etat_bien                         (classification par mots-cles)
  - score_jeune_couple (1-5)          (regles metier + seuils statistiques)
  - justification_couple              (texte template)
  - tags                              (liste calculee)
  - score_marche                      (deja fourni en entree)

Usage autonome :
    python -m scraping.scoring              # enrichit tout (heuristique)
    python -m scraping.scoring --limit 50
"""

from __future__ import annotations

import json
import os
import re
from typing import Optional

from dotenv import load_dotenv
from supabase import create_client

from scraping.enrichissement import (
    calculer_prix_medians,
    score_marche_stat,
    _prix_m2,
)

load_dotenv()

# ---------------------------------------------------------------------------
# Helpers texte
# ---------------------------------------------------------------------------

def _normalise(texte: str) -> str:
    """Minuscules + suppression accents pour comparaison robuste."""
    t = texte.lower()
    for src, dst in [
        ("é", "e"), ("è", "e"), ("ê", "e"), ("ë", "e"),
        ("à", "a"), ("â", "a"), ("ä", "a"),
        ("ù", "u"), ("û", "u"), ("ü", "u"),
        ("î", "i"), ("ï", "i"),
        ("ô", "o"), ("ö", "o"),
        ("ç", "c"),
    ]:
        t = t.replace(src, dst)
    return t


def _contient(texte: str, *mots: str) -> bool:
    t = _normalise(texte)
    return any(m in t for m in mots)


# ---------------------------------------------------------------------------
# Extraction de features depuis le titre
# ---------------------------------------------------------------------------

def extraire_etage(titre: str) -> Optional[int]:
    """Extrait l'etage depuis le titre (ex: '3ème étage', 'étage 2')."""
    t = _normalise(titre)
    for pattern in [
        r"(\d+)\s*(?:eme|er|e)\s+etage",
        r"etage\s*(\d+)",
        r"au\s+(\d+)\s*(?:eme|er|e)",
        r"(\d+)e\s+etage",
    ]:
        m = re.search(pattern, t)
        if m:
            return int(m.group(1))
    if "rez-de-chaussee" in t or "rdc" in t:
        return 0
    return None


def extraire_parking(titre: str) -> bool:
    return _contient(titre, "parking", "garage", "box auto", "box ferm", "place de stat")


def extraire_balcon(titre: str) -> bool:
    return _contient(titre, "balcon", "terrasse", "loggia")


def extraire_vue_mer(titre: str) -> bool:
    t = _normalise(titre)
    return (
        "vue mer" in t
        or "vue sur mer" in t
        or "face mer" in t
        or ("vue" in t and "mer" in t)
        or "panoramique mer" in t
        or "face a la mer" in t
    )


# Mots-cles pour l'etat du bien
_MOTS_NEUF = [
    "neuf", "programme neuf", "livraison", "vefa",
    "rt 2012", "re 2020", "bbc", "etat neuf",
]
_MOTS_BON_ETAT = [
    "bon etat", "tres bon etat", "parfait etat", "bien entretenu",
    "refait", "renove", "renovation recente", "refait a neuf",
    "modernise", "remis a neuf",
]
_MOTS_RAFRAICHIR = [
    "a rafraichir", "quelques travaux", "rafraichissement",
    "leger travaux", "petits travaux",
    "quelques rafraichissements",
]
_MOTS_TRAVAUX = [
    "travaux importants", "gros travaux", "a renover", "a rehabiliter",
    "renovation complete", "entierement a refaire", "a restaurer",
    "remise en etat", "a restructurer",
]


def extraire_etat_bien(titre: str) -> str:
    """
    Retourne 'neuf' | 'bon_etat' | 'a_rafraichir' | 'travaux_importants' | 'inconnu'.
    Ordre de priorite : neuf > travaux_importants > bon_etat > a_rafraichir > inconnu.
    """
    t = _normalise(titre)
    if any(m in t for m in _MOTS_NEUF):
        return "neuf"
    if any(m in t for m in _MOTS_TRAVAUX):
        return "travaux_importants"
    if any(m in t for m in _MOTS_BON_ETAT):
        return "bon_etat"
    if any(m in t for m in _MOTS_RAFRAICHIR):
        return "a_rafraichir"
    if "travaux" in t:
        return "a_rafraichir"
    return "inconnu"


# ---------------------------------------------------------------------------
# Score jeune couple (heuristique)
# ---------------------------------------------------------------------------

# Budget cible jeune couple a Toulon (region PACA)
_BUDGET_IDEAL    = 180_000
_BUDGET_CORRECT  = 250_000
_BUDGET_ELEVE    = 350_000
_BUDGET_EXCESSIF = 450_000

_SURFACE_MIN_COUPLE = 35   # m²
_SURFACE_OK_COUPLE  = 45   # m²


def score_jeune_couple(
    prix: float,
    surface: float,
    pieces: int,
    etat_bien: str,
    score_marche: str,
) -> tuple[int, str]:
    """
    Calcule un score 1-5 pour un jeune couple et genere une justification.

    Returns:
        (score: int, justification: str)
    """
    points: float = 3.0
    raisons_pos: list[str] = []
    raisons_neg: list[str] = []

    # Budget
    if prix > 0:
        if prix <= _BUDGET_IDEAL:
            points += 1.5
            raisons_pos.append(f"prix accessible ({prix:,.0f} €)")
        elif prix <= _BUDGET_CORRECT:
            points += 0.5
            raisons_pos.append(f"prix raisonnable ({prix:,.0f} €)")
        elif prix <= _BUDGET_ELEVE:
            points -= 0.5
            raisons_neg.append(f"budget eleve ({prix:,.0f} €)")
        elif prix <= _BUDGET_EXCESSIF:
            points -= 1.0
            raisons_neg.append(f"budget tres eleve ({prix:,.0f} €)")
        else:
            points -= 2.0
            raisons_neg.append(f"hors budget ({prix:,.0f} €)")

    # Surface
    if surface > 0:
        if surface >= _SURFACE_OK_COUPLE:
            points += 0.5
            raisons_pos.append(f"surface suffisante ({surface:.0f} m²)")
        elif surface < _SURFACE_MIN_COUPLE:
            points -= 1.0
            raisons_neg.append(f"surface insuffisante ({surface:.0f} m²)")

    # Pieces
    if pieces and pieces >= 2:
        points += 0.2
    elif pieces and pieces < 2:
        points -= 0.5
        raisons_neg.append("trop peu de pieces")

    # Etat du bien
    if etat_bien == "neuf":
        points += 0.5
        raisons_pos.append("bien neuf")
    elif etat_bien == "bon_etat":
        points += 0.3
        raisons_pos.append("bon etat general")
    elif etat_bien == "a_rafraichir":
        points -= 0.3
        raisons_neg.append("quelques travaux a prevoir")
    elif etat_bien == "travaux_importants":
        points -= 1.0
        raisons_neg.append("travaux importants")

    # Positionnement marche
    if score_marche == "Opportunite":
        points += 0.5
        raisons_pos.append("bon rapport qualite/prix")
    elif score_marche == "Surevalue":
        points -= 0.5
        raisons_neg.append("prix au-dessus du marche")

    score = max(1, min(5, round(points)))

    pos_txt = ", ".join(raisons_pos)
    neg_txt = ", ".join(raisons_neg)
    if pos_txt and neg_txt:
        justif = f"Points forts : {pos_txt}. Reserves : {neg_txt}."
    elif pos_txt:
        justif = f"Profil adapte : {pos_txt}."
    elif neg_txt:
        justif = f"Peu adapte : {neg_txt}."
    else:
        justif = "Profil neutre, sans critere determinant."

    return score, justif


# ---------------------------------------------------------------------------
# Generation des tags
# ---------------------------------------------------------------------------

_TOUS_LES_TAGS = [
    "vue_mer", "proche_transports", "coup_de_coeur",
    "travaux_importants", "investissement_locatif", "grande_terrasse",
    "parking_inclus", "lumineux", "calme", "centre_ville",
]


def generer_tags(
    titre: str,
    vue_mer: bool,
    parking: bool,
    balcon: bool,
    etat_bien: str,
    score_jeune: int,
    score_marche: str,
    surface: float,
) -> list[str]:
    tags: list[str] = []
    t = _normalise(titre)

    if vue_mer:
        tags.append("vue_mer")

    if parking:
        tags.append("parking_inclus")

    if _contient(titre, "grande terrasse", "terrasse spacieuse", "grande loggia"):
        tags.append("grande_terrasse")

    if etat_bien == "travaux_importants" or "travaux" in t:
        tags.append("travaux_importants")

    if _contient(titre, "lumineux", "clair", "ensoleille", "sud", "plein sud"):
        tags.append("lumineux")

    if _contient(titre, "calme", "tranquille", "sans vis-a-vis"):
        tags.append("calme")

    if _contient(titre, "centre ville", "centre-ville", "hypercentre"):
        tags.append("centre_ville")

    if _contient(titre, "transports", "bus", "tram", "metro", "gare", "proche ecoles"):
        tags.append("proche_transports")

    if score_marche == "Opportunite" and surface >= 25:
        tags.append("investissement_locatif")

    if score_jeune == 5:
        tags.append("coup_de_coeur")

    # Deduplication ordre stable
    seen: set[str] = set()
    result: list[str] = []
    for tag in tags:
        if tag not in seen and tag in _TOUS_LES_TAGS:
            seen.add(tag)
            result.append(tag)

    return result


# ---------------------------------------------------------------------------
# Fonction principale : enrichissement heuristique d'une annonce
# ---------------------------------------------------------------------------

def enrichir_annonce_heuristique(annonce: dict, score_stat: str) -> dict:
    """
    Equivalent de enrichir_annonce_claude() sans appel IA.

    Args:
        annonce:    dict avec au moins titre, prix, surface, pieces
        score_stat: "Opportunite" | "Prix marche" | "Surevalue"

    Returns:
        dict avec les memes cles que enrichir_annonce_claude()
    """
    titre   = annonce.get("titre") or ""
    prix    = float(annonce.get("prix") or 0)
    surface = float(annonce.get("surface") or 0)
    pieces  = int(annonce.get("pieces") or 0)

    etage   = extraire_etage(titre)
    parking = extraire_parking(titre)
    balcon  = extraire_balcon(titre)
    vue_mer = extraire_vue_mer(titre)
    etat    = extraire_etat_bien(titre)

    score, justif = score_jeune_couple(prix, surface, pieces, etat, score_stat)

    tags = generer_tags(titre, vue_mer, parking, balcon, etat, score, score_stat, surface)

    return {
        "score_marche":         score_stat,
        "etage":                etage,
        "parking":              parking,
        "balcon":               balcon,
        "vue_mer":              vue_mer,
        "etat_bien":            etat,
        "score_jeune_couple":   score,
        "justification_couple": justif,
        "tags":                 json.dumps(tags, ensure_ascii=False),
        "resume_ia":            "",
    }


# ---------------------------------------------------------------------------
# Pipeline autonome
# ---------------------------------------------------------------------------

def _supabase():
    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_KEY"]
    return create_client(url, key)


def enrichir_tout_heuristique(limit: Optional[int] = None) -> None:
    print("=" * 60)
    print("  SCORING HEURISTIQUE - Observatoire Immobilier Toulonnais")
    print("=" * 60)

    sb = _supabase()

    print("\n[1/3] Calcul des prix medians par quartier...")
    res_all = sb.table("annonces").select("quartier,prix,surface").execute()
    medianes = calculer_prix_medians(res_all.data or [])
    print(f"  → {len(medianes)} quartiers")

    print("[2/3] Chargement des annonces a enrichir...")
    query = (
        sb.table("annonces")
        .select("id,titre,prix,surface,pieces,quartier,type_bien")
        .is_("score_marche", "null")
        .order("id")
    )
    if limit:
        query = query.limit(limit)
    annonces = query.execute().data or []
    print(f"  → {len(annonces)} annonces")

    if not annonces:
        print("  Rien a enrichir.")
        return

    print("[3/3] Scoring heuristique...\n")
    ok = 0
    for i, annonce in enumerate(annonces, 1):
        pm2        = _prix_m2(annonce)
        quartier   = annonce.get("quartier") or "Inconnu"
        median_q   = medianes.get(quartier, 0)
        score_stat = score_marche_stat(pm2, median_q)

        champs = enrichir_annonce_heuristique(annonce, score_stat)
        sb.table("annonces").update(champs).eq("id", annonce["id"]).execute()
        ok += 1

        print(
            f"  [{i:>4}/{len(annonces)}] {annonce['titre'][:45]:<45} "
            f"| {champs['score_marche']:<12} "
            f"| couple:{champs['score_jeune_couple']} "
            f"| etat:{champs['etat_bien']:<18} "
            f"| tags:{champs['tags']}"
        )

    print(f"\n{'='*60}")
    print(f"  Termine : {ok}/{len(annonces)} annonces enrichies")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# Point d'entree
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Scoring heuristique des annonces")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    enrichir_tout_heuristique(limit=args.limit)
