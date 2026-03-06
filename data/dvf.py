"""
dvf.py — Collecte et nettoyage des données DVF pour Toulon (INSEE 83137)
Projet Observatoire du Marché Immobilier Toulonnais
"""

import json
import urllib.request
import csv
import os
import gzip
import io
import ssl
from datetime import datetime, date

# ─────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────
INSEE_TOULON = "83137"
TYPES_BIENS = {"Appartement", "Maison"}
ANNEES = [2023, 2024]

PRIX_MIN = 30_000
PRIX_MAX = 450_000
SURFACE_MIN = 9
SURFACE_MAX = 500

RAW_DIR = os.path.join(os.path.dirname(__file__), "raw")
os.makedirs(RAW_DIR, exist_ok=True)


# ─────────────────────────────────────────
# 1. COLLECTE
# ─────────────────────────────────────────

def fetch_dvf_year(annee: int) -> list[dict]:
    url = f"https://files.data.gouv.fr/geo-dvf/latest/csv/{annee}/departements/83.csv.gz"
    print(f"  → Téléchargement DVF {annee} (dép. 83)...")
    try:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        with urllib.request.urlopen(url, timeout=60, context=ctx) as response:
            compressed = response.read()

        with gzip.open(io.BytesIO(compressed), 'rt', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = [row for row in reader if row.get("code_commune") == INSEE_TOULON]

        print(f"     {len(rows)} transactions brutes récupérées pour Toulon")
        return rows
    except Exception as e:
        print(f"  ⚠ Erreur téléchargement {annee}: {e}")
        return []


def fetch_all_dvf() -> list[dict]:
    all_rows = []
    for annee in ANNEES:
        rows = fetch_dvf_year(annee)
        raw_path = os.path.join(RAW_DIR, f"dvf_{annee}_raw.json")
        with open(raw_path, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
        print(f"     Sauvegardé : {raw_path}")
        all_rows.extend(rows)
    print(f"\n  Total brut : {len(all_rows)} transactions")
    return all_rows


# ─────────────────────────────────────────
# 2. NETTOYAGE & FILTRAGE
# ─────────────────────────────────────────

def parse_float(val):
    if val is None or val == "":
        return None
    try:
        return float(str(val).replace(",", ".").replace(" ", ""))
    except (ValueError, TypeError):
        return None


def parse_int(val):
    try:
        return int(float(str(val)))
    except (ValueError, TypeError):
        return None


def clean_row(row: dict):
    type_bien = row.get("type_local", "")
    if type_bien not in TYPES_BIENS:
        return None

    prix = parse_float(row.get("valeur_fonciere"))
    surface = parse_float(row.get("surface_reelle_bati"))

    if prix is None or surface is None:
        return None
    if not (PRIX_MIN <= prix <= PRIX_MAX):
        return None
    if not (SURFACE_MIN <= surface <= SURFACE_MAX):
        return None

    prix_m2 = round(prix / surface, 2)
    if prix_m2 < 500 or prix_m2 > 15_000:
        return None

    date_str = row.get("date_mutation", "")
    try:
        date_obj = datetime.strptime(date_str[:10], "%Y-%m-%d").date()
        annee = date_obj.year
        mois = date_obj.month
    except (ValueError, TypeError):
        return None

    pieces = parse_int(row.get("nombre_pieces_principales"))
    code_postal = str(row.get("code_postal", "")).strip()
    quartier = code_postal_to_quartier(code_postal)

    # ── Coordonnées GPS ──
    latitude = parse_float(row.get("latitude"))
    longitude = parse_float(row.get("longitude"))

    return {
        "date": date_str[:10],
        "annee": annee,
        "mois": mois,
        "type_bien": type_bien,
        "surface": surface,
        "prix": prix,
        "prix_m2": prix_m2,
        "pieces": pieces,
        "code_postal": code_postal,
        "quartier": quartier,
        "latitude": latitude if latitude is not None else "",
        "longitude": longitude if longitude is not None else "",
    }


def code_postal_to_quartier(cp: str) -> str:
    mapping = {
        "83000": "Centre-ville",
        "83100": "Ouest (Le Mourillon / Saint-Jean-du-Var)",
        "83200": "Est (La Valette / La Garde)",
    }
    return mapping.get(cp, f"Autre ({cp})")


def clean_all(raw_rows: list[dict]) -> list[dict]:
    cleaned = []
    rejected = 0
    for row in raw_rows:
        result = clean_row(row)
        if result:
            cleaned.append(result)
        else:
            rejected += 1
    print(f"\n  ✔ Lignes valides  : {len(cleaned)}")
    print(f"  ✘ Lignes rejetées : {rejected}")
    return cleaned


# ─────────────────────────────────────────
# 3. EXPORT CSV
# ─────────────────────────────────────────

FIELDNAMES = [
    "date", "annee", "mois", "type_bien", "surface",
    "prix", "prix_m2", "pieces", "code_postal", "quartier",
    "latitude", "longitude"
]


def save_csv(rows: list[dict], path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n  💾 CSV sauvegardé : {path}  ({len(rows)} lignes)")


def load_csv(path: str) -> list[dict]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["surface"] = float(row["surface"])
            row["prix"] = float(row["prix"])
            row["prix_m2"] = float(row["prix_m2"])
            row["annee"] = int(row["annee"])
            row["mois"] = int(row["mois"])
            row["pieces"] = int(row["pieces"]) if row["pieces"] else None
            row["latitude"] = float(row["latitude"]) if row["latitude"] else None
            row["longitude"] = float(row["longitude"]) if row["longitude"] else None
            rows.append(row)
    return rows


# ─────────────────────────────────────────
# 4. STATISTIQUES RAPIDES
# ─────────────────────────────────────────

def quick_stats(rows: list[dict]) -> None:
    if not rows:
        print("  ⚠ Aucune donnée !")
        return

    prix_m2_list = [r["prix_m2"] for r in rows]
    prix_m2_moy = sum(prix_m2_list) / len(prix_m2_list)
    prix_m2_list_sorted = sorted(prix_m2_list)
    n = len(prix_m2_list_sorted)
    mediane = prix_m2_list_sorted[n // 2]

    apparts = [r for r in rows if r["type_bien"] == "Appartement"]
    maisons = [r for r in rows if r["type_bien"] == "Maison"]
    avec_gps = [r for r in rows if r["latitude"] and r["longitude"]]

    print("\n── Statistiques rapides ──────────────────────")
    print(f"  Transactions totales : {len(rows)}")
    print(f"  Appartements         : {len(apparts)}")
    print(f"  Maisons              : {len(maisons)}")
    print(f"  Avec coordonnées GPS : {len(avec_gps)}")
    print(f"  Prix/m² moyen        : {prix_m2_moy:.0f} €/m²")
    print(f"  Prix/m² médian       : {mediane:.0f} €/m²")
    print(f"  Prix/m² min          : {min(prix_m2_list):.0f} €/m²")
    print(f"  Prix/m² max          : {max(prix_m2_list):.0f} €/m²")

    quartiers = {}
    for r in rows:
        q = r["quartier"]
        quartiers.setdefault(q, []).append(r["prix_m2"])
    print("\n  Par quartier :")
    for q, vals in sorted(quartiers.items()):
        moy = sum(vals) / len(vals)
        print(f"    {q:<45} {len(vals):>4} tx  |  {moy:.0f} €/m²")
    print("──────────────────────────────────────────────")


# ─────────────────────────────────────────
# 5. PIPELINE PRINCIPAL
# ─────────────────────────────────────────

def run_pipeline(output_csv: str = None) -> list[dict]:
    if output_csv is None:
        output_csv = os.path.join(RAW_DIR, "dvf_clean.csv")

    print("═" * 50)
    print("  PIPELINE DVF — Toulon (83137)")
    print("═" * 50)

    raw = fetch_all_dvf()
    clean = clean_all(raw)

    if len(clean) < 100:
        print(f"\n  ⚠ Attention : seulement {len(clean)} transactions valides.")

    save_csv(clean, output_csv)
    quick_stats(clean)

    return clean


if __name__ == "__main__":
    run_pipeline()
