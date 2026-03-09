from __future__ import annotations
"""
Observatoire Immobilier Toulonnais
Collecte des annonces immobilières via l'API publique Bienici.com

Bienici agrège les annonces de toutes les agences toulonnaises
(Century21, Orpi, Laforêt, Nexity, ERA, particuliers…).
Le champ `source` identifie l'agence ou le type de vendeur.
"""

import csv
import json
import os
import time
import urllib.parse
import urllib.request

from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

BIENICI_API    = "https://www.bienici.com/realEstateAds.json"
TOULON_ZONE_ID = "-35280"
PAGE_SIZE      = 24
MAX_PAGES      = 60   # 60 × 24 = 1 440 annonces max (total dispo : ~1437)
BATCH_SIZE     = 100
MAX_RETRIES    = 3
RETRY_DELAYS   = [2, 5, 15]

CSV_COLUMNS = [
    "titre", "prix", "surface", "nb_pieces",
    "quartier", "type_bien", "prix_m2", "url", "source",
]

CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "annonces.csv")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
    "Accept-Language": "fr-FR,fr;q=0.9",
    "Referer": "https://www.bienici.com/",
}

TYPE_BIEN_MAP = {
    "flat":       "appartement",
    "house":      "maison",
    "loft":       "appartement",
    "studio":     "appartement",
    "duplex":     "appartement",
    "townhouse":  "maison",
    "programme":  "appartement",  # programme neuf
    # exclusions explicites
    "parking":    None,
    "land":       None,
    "terrain":    None,
    "commercial": None,
    "office":     None,
}


# ---------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------

def _scalar(val, default=None):
    """Extrait le premier élément si val est une liste, sinon retourne val."""
    if isinstance(val, list):
        return val[0] if val else default
    return val if val is not None else default


def _mediane(valeurs: list[float]) -> float:
    """Calcule la médiane en Python pur — pas de numpy/statistics."""
    if not valeurs:
        return 0.0
    triees = sorted(valeurs)
    n = len(triees)
    mid = n // 2
    if n % 2 == 1:
        return triees[mid]
    return (triees[mid - 1] + triees[mid]) / 2.0


# ---------------------------------------------------------------------------
# Collecte
# ---------------------------------------------------------------------------

def _build_url(page: int) -> str:
    filters = {
        "size":           PAGE_SIZE,
        "from":           page * PAGE_SIZE,
        "filterType":     "buy",
        "onTheMarket":    [True],
        "zoneIdsByTypes": {"zoneIds": [TOULON_ZONE_ID]},
        "maxPrice":       500000,
        "propertyType":   ["house", "flat"],
    }
    return BIENICI_API + "?filters=" + urllib.parse.quote(json.dumps(filters))


def scrape_bienici(max_pages: int = MAX_PAGES) -> list[dict]:
    """
    Collecte toutes les annonces disponibles sur Bienici pour Toulon.
    La source est l'agence ou le type de vendeur (accountDisplayName / accountType).

    Args:
        max_pages: Nombre de pages (24 annonces/page).

    Returns:
        Liste de dicts d'annonces brutes.
    """
    annonces = []

    for page in range(max_pages):
        url = _build_url(page)
        req = urllib.request.Request(url, headers=HEADERS)

        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            print(f"  [Bienici] Erreur page {page + 1} : {exc}")
            continue

        ads = data.get("realEstateAds", [])
        if not ads:
            print(f"  [Bienici] Page {page + 1} vide — arrêt.")
            break

        for ad in ads:
            try:
                prix          = _scalar(ad.get("price"))
                surface       = _scalar(ad.get("surfaceArea"))
                rooms         = _scalar(ad.get("roomsQuantity"), default=0)
                property_type = _scalar(ad.get("propertyType"), default="")
                titre         = _scalar(ad.get("title"), default="")
                account_name  = _scalar(ad.get("accountDisplayName"), default="")
                account_type  = _scalar(ad.get("accountType"), default="")
                annonce_id    = _scalar(ad.get("id"), default="")

                type_bien = TYPE_BIEN_MAP.get(property_type, property_type)

                district_raw = _scalar(ad.get("district")) or {}
                if isinstance(district_raw, dict):
                    quartier = district_raw.get("libelle") or ad.get("city") or "Toulon"
                else:
                    quartier = ad.get("city") or "Toulon"

                annonce_url = f"https://www.bienici.com/annonce/{annonce_id}"

                # Source = nom de l'agence si disponible, sinon type de compte
                if account_name:
                    source = account_name
                elif account_type == "owner":
                    source = "Particulier"
                elif account_type == "mandatary":
                    source = "Mandataire"
                else:
                    source = "Agence"

                annonces.append({
                    "titre":     titre,
                    "prix":      int(prix) if prix is not None else 0,
                    "surface":   float(surface) if surface is not None else 0.0,
                    "nb_pieces": int(rooms) if rooms else 0,
                    "quartier":  quartier,
                    "type_bien": type_bien,
                    "url":       annonce_url,
                    "source":    source,
                })

            except Exception as exc:
                print(f"  [Bienici] Erreur extraction annonce : {exc}")
                continue

        print(f"  [Bienici] Page {page + 1}/{max_pages} — {len(ads)} annonces")
        time.sleep(0.5)

    return annonces


# ---------------------------------------------------------------------------
# Nettoyage
# ---------------------------------------------------------------------------

def nettoyer(annonces: list[dict]) -> list[dict]:
    """
    Filtre, déduplique et calcule prix_m2.
    Règles :
      - prix > 0 && <= 500 000 €
      - surface > 0
      - type_bien connu et non exclu (not None)
      - url non vide et ne se terminant pas par '/annonce/'
      - déduplication sur URL
      - déduplication sur empreinte (prix, surface, titre[:30])
    """
    vues_url        = set()
    vues_empreinte  = set()
    resultats       = []

    # Compteurs de rejet
    rejets = {
        "prix_absent":        0,
        "prix_hors_budget":   0,
        "surface_absente":    0,
        "type_exclu":         0,
        "url_invalide":       0,
        "doublon_url":        0,
        "doublon_empreinte":  0,
    }

    for ann in annonces:
        prix      = ann.get("prix", 0)
        surface   = ann.get("surface", 0.0)
        url       = ann.get("url", "")
        type_bien = ann.get("type_bien")
        titre     = ann.get("titre", "")

        # --- Validation prix ---
        if prix is None or prix == 0:
            rejets["prix_absent"] += 1
            continue
        if prix > 500_000:
            rejets["prix_hors_budget"] += 1
            continue

        # --- Validation surface ---
        if surface is None or surface == 0:
            rejets["surface_absente"] += 1
            continue

        # --- Validation type_bien ---
        # type_bien vaut None si c'est un type exclu (parking, terrain, etc.)
        if type_bien is None:
            rejets["type_exclu"] += 1
            continue

        # --- Validation URL ---
        if not url or url.endswith("/annonce/"):
            rejets["url_invalide"] += 1
            continue

        # --- Déduplication URL ---
        if url in vues_url:
            rejets["doublon_url"] += 1
            continue

        # --- Déduplication empreinte ---
        empreinte = (prix, surface, titre[:30])
        if empreinte in vues_empreinte:
            rejets["doublon_empreinte"] += 1
            continue

        vues_url.add(url)
        vues_empreinte.add(empreinte)

        prix_m2 = round(prix / surface, 2)  # arithmétique pure

        resultats.append({
            "titre":     titre,
            "prix":      prix,
            "surface":   surface,
            "nb_pieces": ann.get("nb_pieces", 0),
            "quartier":  ann.get("quartier", ""),
            "type_bien": type_bien,
            "prix_m2":   prix_m2,
            "url":       url,
            "source":    ann.get("source", ""),
        })

    # --- Logging des rejets ---
    total_rejet = sum(rejets.values())
    print(f"\n[Nettoyage] {len(annonces)} brutes → {len(resultats)} nettes "
          f"({total_rejet} rejetées)")
    for motif, nb in rejets.items():
        if nb:
            print(f"  {motif:<22} : {nb:>4}")

    return resultats


# ---------------------------------------------------------------------------
# Sauvegarde CSV
# ---------------------------------------------------------------------------

def sauvegarder_csv(annonces: list[dict]) -> None:
    """Écrit data/annonces.csv — colonnes dans l'ordre exact attendu par le CI."""
    chemin = os.path.abspath(CSV_PATH)
    os.makedirs(os.path.dirname(chemin), exist_ok=True)

    with open(chemin, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(annonces)

    print(f"[CSV] {len(annonces)} annonces → {chemin}")


# ---------------------------------------------------------------------------
# Envoi Supabase
# ---------------------------------------------------------------------------

def envoyer_supabase(annonces: list[dict]) -> None:
    """Upsert vers Supabase table `annonces`. Mapping : url→lien, nb_pieces→pieces.
    - batch_size = BATCH_SIZE (100)
    - Retry 3 fois avec backoff RETRY_DELAYS
    - Vérification count post-upsert
    """
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")

    if not supabase_url or not supabase_key:
        print("[Supabase] Credentials manquants — envoi ignoré.")
        return

    client = create_client(supabase_url, supabase_key)

    lignes = [
        {
            "titre":     a.get("titre", ""),
            "prix":      a.get("prix", 0),
            "surface":   a.get("surface", 0.0),
            "pieces":    a.get("nb_pieces", 0),
            "quartier":  a.get("quartier", ""),
            "type_bien": a.get("type_bien", ""),
            "lien":      a.get("url", ""),
            "source":    a.get("source", ""),
        }
        for a in annonces
        if a.get("url")
    ]

    total_envoye = 0

    for i in range(0, len(lignes), BATCH_SIZE):
        batch      = lignes[i : i + BATCH_SIZE]
        batch_num  = i // BATCH_SIZE + 1
        success    = False

        for tentative in range(MAX_RETRIES):
            try:
                client.table("annonces").upsert(batch, on_conflict="lien").execute()
                total_envoye += len(batch)
                print(f"  [Supabase] Batch {batch_num} : {len(batch)} envoyées")
                success = True
                break
            except Exception as exc:
                delai = RETRY_DELAYS[tentative] if tentative < len(RETRY_DELAYS) else RETRY_DELAYS[-1]
                print(
                    f"  [Supabase] Erreur batch {batch_num} "
                    f"(tentative {tentative + 1}/{MAX_RETRIES}) : {exc} "
                    f"— retry dans {delai}s"
                )
                if tentative < MAX_RETRIES - 1:
                    time.sleep(delai)

        if not success:
            print(f"  [Supabase] Batch {batch_num} abandonné après {MAX_RETRIES} tentatives.")

    print(f"[Supabase] Total : {total_envoye} annonces envoyées")

    # --- Vérification count post-upsert ---
    try:
        result = (
            client.table("annonces")
            .select("*", count="exact")
            .execute()
        )
        count = result.count if hasattr(result, "count") else "?"
        print(f"[Supabase] Vérification : {count} lignes dans la table 'annonces'")
    except Exception as exc:
        print(f"[Supabase] Impossible de vérifier le count : {exc}")


# ---------------------------------------------------------------------------
# Rapport qualité
# ---------------------------------------------------------------------------

def rapport_qualite(brutes: list[dict], nettes: list[dict], rejets_detail: dict | None = None) -> None:
    """Affiche un rapport qualité complet après nettoyage."""
    nb_brut  = len(brutes)
    nb_net   = len(nettes)
    nb_perdu = nb_brut - nb_net

    print("\n" + "=" * 55)
    print("RAPPORT QUALITE")
    print("=" * 55)

    # --- Volume ---
    print(f"\n[Volume]")
    print(f"  Brut    : {nb_brut:>6}")
    print(f"  Perdu   : {nb_perdu:>6}")
    if rejets_detail:
        for motif, nb in rejets_detail.items():
            if nb:
                print(f"    {motif:<22} : {nb:>4}")
    print(f"  Net     : {nb_net:>6}")

    if not nettes:
        print("[Rapport] Aucune annonce nette à analyser.")
        return

    # --- Top 10 quartiers ---
    quartiers: dict[str, int] = {}
    for a in nettes:
        q = a.get("quartier") or "Inconnu"
        quartiers[q] = quartiers.get(q, 0) + 1
    print(f"\n[Top 10 quartiers]")
    for q, nb in sorted(quartiers.items(), key=lambda x: -x[1])[:10]:
        print(f"  {q:<35} : {nb:>4}")

    # --- Distribution type_bien ---
    types: dict[str, int] = {}
    for a in nettes:
        t = a.get("type_bien") or "Inconnu"
        types[t] = types.get(t, 0) + 1
    print(f"\n[Distribution type_bien]")
    for t, nb in sorted(types.items(), key=lambda x: -x[1]):
        print(f"  {t:<20} : {nb:>4}")

    # --- Stats prix ---
    prix_liste = [a["prix"] for a in nettes if a.get("prix")]
    if prix_liste:
        print(f"\n[Stats prix]")
        print(f"  Min    : {min(prix_liste):>10,.0f} €")
        print(f"  Max    : {max(prix_liste):>10,.0f} €")
        print(f"  Median : {_mediane(prix_liste):>10,.0f} €")

    # --- Stats prix_m2 ---
    pm2_liste = [a["prix_m2"] for a in nettes if a.get("prix_m2")]
    if pm2_liste:
        print(f"\n[Stats prix_m2]")
        print(f"  Min    : {min(pm2_liste):>10,.0f} €/m²")
        print(f"  Max    : {max(pm2_liste):>10,.0f} €/m²")
        print(f"  Median : {_mediane(pm2_liste):>10,.0f} €/m²")

    # --- Sources ---
    sources: dict[str, int] = {}
    for a in nettes:
        s = a.get("source") or "Inconnu"
        sources[s] = sources.get(s, 0) + 1
    print(f"\n[Sources] {len(sources)} distinctes — Top 5 :")
    for src, nb in sorted(sources.items(), key=lambda x: -x[1])[:5]:
        print(f"  {src:<35} : {nb:>4} annonces")

    print("=" * 55)


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Observatoire Immobilier Toulonnais — Scraping ===\n")

    print(f"--- Collecte Bienici ({MAX_PAGES} pages × {PAGE_SIZE} = ~{MAX_PAGES * PAGE_SIZE}) ---")
    brutes = scrape_bienici(max_pages=MAX_PAGES)
    print(f"\n[Brut] {len(brutes)} annonces collectées")

    print("\n--- Nettoyage & déduplication ---")

    # On instrumente nettoyer pour capturer les rejets
    # (réutilisation du log interne de nettoyer)
    nettes = nettoyer(brutes)

    # Rapport qualité complet
    rapport_qualite(brutes, nettes)

    print("\n--- Sauvegarde CSV ---")
    sauvegarder_csv(nettes)

    print("\n--- Envoi Supabase ---")
    envoyer_supabase(nettes)

    print(f"\n=== Terminé : {len(nettes)} annonces dans data/annonces.csv ===")
