"""
Observatoire Immobilier Toulonnais
Scraping des annonces immobilières depuis PAP.fr et Logic-Immo.com
"""

import os
import re
import time
import csv

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from supabase import create_client

# Chargement des variables d'environnement
load_dotenv()

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "fr-FR,fr;q=0.9,en-US;q=0.8,en;q=0.7",
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;"
        "q=0.9,image/webp,*/*;q=0.8"
    ),
    "Connection": "keep-alive",
}

CSV_COLUMNS = [
    "titre",
    "prix",
    "surface",
    "nb_pieces",
    "quartier",
    "type_bien",
    "prix_m2",
    "url",
    "source",
]

CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "annonces.csv")


# ---------------------------------------------------------------------------
# Utilitaires d'extraction
# ---------------------------------------------------------------------------

def _extraire_nombre(texte: str) -> float | None:
    """
    Extrait le premier nombre (entier ou décimal) contenu dans une chaîne.
    Supprime les séparateurs de milliers (espace, point) avant de parser.
    Retourne None si aucun nombre trouvé.
    """
    if not texte:
        return None
    # Suppression des espaces insécables et ordinaires utilisés comme séparateurs de milliers
    texte_nettoye = texte.replace("\xa0", "").replace(" ", "").replace(".", "").replace(",", ".")
    match = re.search(r"\d+(?:\.\d+)?", texte_nettoye)
    if match:
        return float(match.group())
    return None


# ---------------------------------------------------------------------------
# Scraping PAP.fr
# ---------------------------------------------------------------------------

def scrape_pap(nb_pages: int = 10) -> list[dict]:
    """
    Scrape les annonces de vente immobilière à Toulon sur PAP.fr.

    Args:
        nb_pages: Nombre de pages à parcourir (défaut : 10).

    Returns:
        Liste de dicts avec les clés :
        titre, prix, surface, nb_pieces, quartier, type_bien, url, source.
    """
    annonces = []
    base_url = "https://www.pap.fr/annonce/vente-immobiliere-toulon-83000-g43624"

    for page in range(1, nb_pages + 1):
        url = f"{base_url}?page={page}"
        try:
            response = requests.get(url, headers=HEADERS, timeout=15)
            response.raise_for_status()
        except requests.RequestException as exc:
            print(f"[PAP] Erreur page {page} : {exc}")
            continue  # On continue avec la page suivante

        soup = BeautifulSoup(response.text, "lxml")

        # Chaque annonce est dans un article ou une card — sélecteur à adapter
        # selon la structure réelle du site au moment du scraping
        cards = soup.select("a.search-list-item")
        if not cards:
            # Sélecteur alternatif si la structure a changé
            cards = soup.select("div.annonce")

        for card in cards:
            try:
                # Titre / type de bien
                titre_el = card.select_one(".item-type, .annonce-title, h2, h3")
                titre = titre_el.get_text(strip=True) if titre_el else ""

                # Type de bien (souvent inclus dans le titre)
                type_bien = titre.split()[0] if titre else ""

                # Prix
                prix_el = card.select_one(".item-price, .price, [class*='prix']")
                prix_texte = prix_el.get_text(strip=True) if prix_el else ""
                prix_val = _extraire_nombre(prix_texte)
                prix = int(prix_val) if prix_val is not None else 0

                # Surface
                surface_el = card.select_one(
                    "[class*='surface'], [class*='area'], .item-tags"
                )
                surface_texte = surface_el.get_text(strip=True) if surface_el else ""
                surface_match = re.search(r"(\d+(?:[.,]\d+)?)\s*m", surface_texte)
                surface = float(surface_match.group(1).replace(",", ".")) if surface_match else 0.0

                # Nombre de pièces
                pieces_match = re.search(r"(\d+)\s*p[iè]", surface_texte, re.IGNORECASE)
                nb_pieces = int(pieces_match.group(1)) if pieces_match else 0

                # Quartier
                lieu_el = card.select_one(
                    ".item-location, .location, [class*='ville'], [class*='quartier']"
                )
                quartier = lieu_el.get_text(strip=True) if lieu_el else "Toulon"

                # URL de l'annonce
                href = card.get("href", "")
                annonce_url = (
                    f"https://www.pap.fr{href}" if href.startswith("/") else href
                )

                annonces.append(
                    {
                        "titre": titre,
                        "prix": prix,
                        "surface": surface,
                        "nb_pieces": nb_pieces,
                        "quartier": quartier,
                        "type_bien": type_bien,
                        "url": annonce_url,
                        "source": "PAP",
                    }
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[PAP] Erreur extraction annonce : {exc}")
                continue

        print(f"[PAP] Page {page}/{nb_pages} — {len(cards)} annonces trouvées")
        time.sleep(2)  # Délai poli entre les requêtes

    return annonces


# ---------------------------------------------------------------------------
# Scraping Logic-Immo.com
# ---------------------------------------------------------------------------

def scrape_logic_immo(nb_pages: int = 10) -> list[dict]:
    """
    Scrape les annonces de vente immobilière à Toulon sur Logic-Immo.com.

    Args:
        nb_pages: Nombre de pages à parcourir (défaut : 10).

    Returns:
        Liste de dicts avec les clés :
        titre, prix, surface, nb_pieces, quartier, type_bien, url, source.
    """
    annonces = []
    base_url = (
        "https://www.logic-immo.com/vente-immobilier-toulon,83000-84+83100-84"
        "/options/groupprptypesids=1,2,6,7,13/page/{n}"
    )

    for page in range(1, nb_pages + 1):
        url = base_url.format(n=page)
        try:
            response = requests.get(url, headers=HEADERS, timeout=15)
            response.raise_for_status()
        except requests.RequestException as exc:
            print(f"[Logic-Immo] Erreur page {page} : {exc}")
            continue

        soup = BeautifulSoup(response.text, "lxml")

        # Sélecteurs courants de Logic-Immo (peuvent évoluer)
        cards = soup.select("article.offer-card, div.offer-block, div[class*='offer']")
        if not cards:
            cards = soup.select("li.offer-item")

        for card in cards:
            try:
                # Titre
                titre_el = card.select_one(
                    "h2.offer-card-title, h2, h3, [class*='title']"
                )
                titre = titre_el.get_text(strip=True) if titre_el else ""

                # Type de bien
                type_el = card.select_one("[class*='type'], [class*='category']")
                type_bien = type_el.get_text(strip=True) if type_el else titre.split()[0] if titre else ""

                # Prix
                prix_el = card.select_one(
                    "[class*='price'], [class*='prix'], span.price"
                )
                prix_texte = prix_el.get_text(strip=True) if prix_el else ""
                prix_val = _extraire_nombre(prix_texte)
                prix = int(prix_val) if prix_val is not None else 0

                # Surface & pièces (souvent dans un bloc de caractéristiques)
                specs_el = card.select_one(
                    "[class*='criteria'], [class*='specs'], [class*='features']"
                )
                specs_texte = specs_el.get_text(strip=True) if specs_el else ""

                surface_match = re.search(r"(\d+(?:[.,]\d+)?)\s*m", specs_texte)
                surface = float(surface_match.group(1).replace(",", ".")) if surface_match else 0.0

                pieces_match = re.search(r"(\d+)\s*p[iè]", specs_texte, re.IGNORECASE)
                nb_pieces = int(pieces_match.group(1)) if pieces_match else 0

                # Quartier
                lieu_el = card.select_one(
                    "[class*='location'], [class*='city'], [class*='localisation']"
                )
                quartier = lieu_el.get_text(strip=True) if lieu_el else "Toulon"

                # URL
                lien_el = card.select_one("a[href]")
                href = lien_el.get("href", "") if lien_el else ""
                annonce_url = (
                    f"https://www.logic-immo.com{href}"
                    if href.startswith("/")
                    else href
                )

                annonces.append(
                    {
                        "titre": titre,
                        "prix": prix,
                        "surface": surface,
                        "nb_pieces": nb_pieces,
                        "quartier": quartier,
                        "type_bien": type_bien,
                        "url": annonce_url,
                        "source": "Logic-Immo",
                    }
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[Logic-Immo] Erreur extraction annonce : {exc}")
                continue

        print(f"[Logic-Immo] Page {page}/{nb_pages} — {len(cards)} annonces trouvées")
        time.sleep(2)

    return annonces


# ---------------------------------------------------------------------------
# Nettoyage
# ---------------------------------------------------------------------------

def nettoyer(annonces: list[dict]) -> list[dict]:
    """
    Filtre et nettoie la liste d'annonces :
    - Garde uniquement les annonces avec prix > 0, <= 500 000 €
    - Garde uniquement les annonces avec surface > 0
    - Calcule le prix au m² (prix_m2)
    - Déduplique sur l'URL

    Args:
        annonces: Liste brute de dicts d'annonces.

    Returns:
        Liste nettoyée et dédupliquée.
    """
    vues = set()
    resultats = []

    for ann in annonces:
        prix = ann.get("prix", 0)
        surface = ann.get("surface", 0.0)
        url = ann.get("url", "")

        # Filtres de qualité
        if prix <= 0 or prix > 500_000:
            continue
        if surface <= 0:
            continue
        if not url or url in vues:
            continue

        vues.add(url)

        # Calcul du prix au m² (pas de numpy/sklearn, arithmétique pure)
        prix_m2 = round(prix / surface, 2)

        resultats.append(
            {
                "titre": ann.get("titre", ""),
                "prix": prix,
                "surface": surface,
                "nb_pieces": ann.get("nb_pieces", 0),
                "quartier": ann.get("quartier", ""),
                "type_bien": ann.get("type_bien", ""),
                "prix_m2": prix_m2,
                "url": url,
                "source": ann.get("source", ""),
            }
        )

    print(f"[Nettoyage] {len(resultats)} annonces retenues sur {len(annonces)} brutes")
    return resultats


# ---------------------------------------------------------------------------
# Sauvegarde CSV
# ---------------------------------------------------------------------------

def sauvegarder_csv(annonces: list[dict]) -> None:
    """
    Sauvegarde les annonces dans data/annonces.csv avec les colonnes dans
    l'ordre exact défini par CSV_COLUMNS.

    Args:
        annonces: Liste de dicts d'annonces nettoyées.
    """
    chemin = os.path.abspath(CSV_PATH)
    os.makedirs(os.path.dirname(chemin), exist_ok=True)

    with open(chemin, mode="w", newline="", encoding="utf-8") as fichier:
        writer = csv.DictWriter(
            fichier,
            fieldnames=CSV_COLUMNS,
            extrasaction="ignore",  # Ignore les clés inconnues
        )
        writer.writeheader()
        writer.writerows(annonces)

    print(f"[CSV] {len(annonces)} annonces sauvegardées dans {chemin}")


# ---------------------------------------------------------------------------
# Envoi vers Supabase
# ---------------------------------------------------------------------------

def envoyer_supabase(annonces: list[dict]) -> None:
    """
    Envoie les annonces vers la table Supabase `annonces` via upsert.
    Les clés sont remappées : url → lien, nb_pieces → pieces.

    Les credentials Supabase sont lus depuis les variables d'environnement
    SUPABASE_URL et SUPABASE_KEY (fichier .env).

    Args:
        annonces: Liste de dicts d'annonces nettoyées.
    """
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")

    if not supabase_url or not supabase_key:
        print("[Supabase] SUPABASE_URL ou SUPABASE_KEY manquant — envoi ignoré.")
        return

    client = create_client(supabase_url, supabase_key)

    # Remappage des colonnes pour correspondre au schéma Supabase
    lignes = []
    for ann in annonces:
        lignes.append(
            {
                "titre": ann.get("titre", ""),
                "prix": ann.get("prix", 0),
                "surface": ann.get("surface", 0.0),
                "pieces": ann.get("nb_pieces", 0),          # nb_pieces → pieces
                "quartier": ann.get("quartier", ""),
                "type_bien": ann.get("type_bien", ""),
                "prix_m2": ann.get("prix_m2", 0.0),
                "lien": ann.get("url", ""),                  # url → lien
                "source": ann.get("source", ""),
            }
        )

    try:
        # Upsert avec lien comme clé de conflit pour éviter les doublons
        result = (
            client.table("annonces")
            .upsert(lignes, on_conflict="lien")
            .execute()
        )
        print(f"[Supabase] {len(lignes)} annonces envoyées avec succès.")
        return result
    except Exception as exc:  # noqa: BLE001
        print(f"[Supabase] Erreur lors de l'envoi : {exc}")


# ---------------------------------------------------------------------------
# Point d'entrée principal
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Observatoire Immobilier Toulonnais — Scraping ===\n")

    # 1. Scraping des sources
    print("--- Scraping PAP.fr ---")
    annonces_pap = scrape_pap(nb_pages=10)

    print("\n--- Scraping Logic-Immo.com ---")
    annonces_logic = scrape_logic_immo(nb_pages=10)

    # 2. Fusion des deux sources
    toutes_annonces = annonces_pap + annonces_logic
    print(f"\n[Fusion] {len(toutes_annonces)} annonces brutes au total")

    # 3. Nettoyage & déduplication
    print("\n--- Nettoyage ---")
    annonces_nettes = nettoyer(toutes_annonces)

    # 4. Sauvegarde CSV
    print("\n--- Sauvegarde CSV ---")
    sauvegarder_csv(annonces_nettes)

    # 5. Envoi vers Supabase
    print("\n--- Envoi Supabase ---")
    envoyer_supabase(annonces_nettes)

    print("\n=== Scraping terminé ===")
