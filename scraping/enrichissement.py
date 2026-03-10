"""
Enrichissement IA des annonces immobilieres via Claude API (Haiku).
- Scoring marche : Opportunite / Prix marche / Surevalue
- Extraction structuree : etage, parking, balcon, vue mer, etat
- Score jeune couple (1-5) avec justification
- Tags automatiques

Usage :
    python -m scraping.enrichissement              # enrichit tout
    python -m scraping.enrichissement --limit 50   # test sur 50 annonces
"""

from __future__ import annotations

import json
import os
import time
import csv
import re
from typing import Optional

import anthropic
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

# ---------------------------------------------------------------------------
# Clients
# ---------------------------------------------------------------------------

def _supabase():
    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_KEY"]
    return create_client(url, key)


def _anthropic():
    return anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


# ---------------------------------------------------------------------------
# Prix median par quartier (calcul pur Python, sans numpy)
# ---------------------------------------------------------------------------

def _mediane(valeurs: list[float]) -> float:
    s = sorted(v for v in valeurs if v and v > 0)
    if not s:
        return 0.0
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 == 1 else (s[mid - 1] + s[mid]) / 2.0


def _prix_m2(annonce: dict) -> float:
    """Calcule le prix/m2 depuis prix et surface."""
    try:
        prix = float(annonce.get("prix") or 0)
        surface = float(annonce.get("surface") or 0)
        return round(prix / surface, 2) if surface > 0 else 0.0
    except (TypeError, ValueError):
        return 0.0


def calculer_prix_medians(annonces: list[dict]) -> dict[str, float]:
    """Retourne {quartier: prix_m2_median}."""
    par_quartier: dict[str, list[float]] = {}
    for a in annonces:
        q = a.get("quartier") or "Inconnu"
        pm2 = _prix_m2(a)
        if pm2 > 0:
            par_quartier.setdefault(q, []).append(pm2)
    return {q: _mediane(vals) for q, vals in par_quartier.items()}


# ---------------------------------------------------------------------------
# Scoring statistique du marche
# ---------------------------------------------------------------------------

def score_marche_stat(prix_m2: float, median_quartier: float) -> str:
    """
    Retourne 'Opportunite', 'Prix marche' ou 'Surevalue'
    selon l'ecart au prix median du quartier.
    """
    if not median_quartier or median_quartier <= 0:
        return "Prix marche"
    ratio = prix_m2 / median_quartier
    if ratio < 0.90:
        return "Opportunite"
    elif ratio > 1.10:
        return "Surevalue"
    else:
        return "Prix marche"


# ---------------------------------------------------------------------------
# Prompt Claude
# ---------------------------------------------------------------------------

PROMPT_SYSTEME = """Tu es un expert immobilier specialise dans le marche toulonnais.
Analyse l'annonce immobiliere fournie et reponds UNIQUEMENT avec un objet JSON valide,
sans texte supplementaire avant ou apres le JSON.

Le JSON doit avoir exactement cette structure :
{
  "etage": null ou entier (ex: 2),
  "parking": true ou false,
  "balcon": true ou false,
  "vue_mer": true ou false,
  "etat_bien": "neuf" | "bon_etat" | "a_rafraichir" | "travaux_importants" | "inconnu",
  "score_jeune_couple": entier entre 1 et 5,
  "justification_couple": chaine courte (max 80 mots) expliquant le score,
  "tags": liste de tags parmi ["vue_mer","proche_transports","coup_de_coeur",
    "travaux_importants","investissement_locatif","grande_terrasse",
    "parking_inclus","lumineux","calme","centre_ville"],
  "resume_ia": chaine courte (max 60 mots) resumant les points cles de l'annonce
}

Criteres score_jeune_couple :
  5 = ideal (budget accessible, bien place, peu de travaux, espace suffisant)
  4 = tres bien
  3 = convenable
  2 = peu adapte
  1 = inadapte (trop cher, trop petit, trop de travaux)
"""


def enrichir_annonce_claude(
    client: anthropic.Anthropic,
    annonce: dict,
    score_stat: str,
) -> dict:
    """
    Appelle Claude Haiku pour enrichir une annonce.
    Retourne un dict avec les champs enrichis.
    """
    titre = annonce.get("titre", "")
    prix = annonce.get("prix", "")
    surface = annonce.get("surface", "")
    nb_pieces = annonce.get("pieces", "")  # colonne Supabase = "pieces"
    quartier = annonce.get("quartier", "")
    type_bien = annonce.get("type_bien", "")
    pm2 = _prix_m2(annonce)

    contenu_annonce = (
        f"Titre : {titre}\n"
        f"Type : {type_bien}\n"
        f"Quartier : {quartier} (Toulon)\n"
        f"Prix : {prix} €\n"
        f"Surface : {surface} m²\n"
        f"Pieces : {nb_pieces}\n"
        f"Prix/m² : {pm2} €/m²\n"
        f"Positionnement marche (statistique) : {score_stat}\n"
    )

    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        system=PROMPT_SYSTEME,
        messages=[{"role": "user", "content": contenu_annonce}],
    )

    raw = message.content[0].text.strip()

    # Nettoyage si le modele enveloppe dans ```json ... ```
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = {}

    return {
        "score_marche": score_stat,
        "etage": data.get("etage"),
        "parking": data.get("parking"),
        "balcon": data.get("balcon"),
        "vue_mer": data.get("vue_mer"),
        "etat_bien": data.get("etat_bien", "inconnu"),
        "score_jeune_couple": data.get("score_jeune_couple"),
        "justification_couple": data.get("justification_couple", ""),
        "tags": json.dumps(data.get("tags", []), ensure_ascii=False),
        "resume_ia": data.get("resume_ia", ""),
    }


# ---------------------------------------------------------------------------
# Migration Supabase : ajout des colonnes si elles n'existent pas
# ---------------------------------------------------------------------------

NOUVELLES_COLONNES_SQL = """
ALTER TABLE annonces
  ADD COLUMN IF NOT EXISTS score_marche TEXT,
  ADD COLUMN IF NOT EXISTS etage INTEGER,
  ADD COLUMN IF NOT EXISTS parking BOOLEAN,
  ADD COLUMN IF NOT EXISTS balcon BOOLEAN,
  ADD COLUMN IF NOT EXISTS vue_mer BOOLEAN,
  ADD COLUMN IF NOT EXISTS etat_bien TEXT,
  ADD COLUMN IF NOT EXISTS score_jeune_couple INTEGER,
  ADD COLUMN IF NOT EXISTS justification_couple TEXT,
  ADD COLUMN IF NOT EXISTS tags TEXT,
  ADD COLUMN IF NOT EXISTS resume_ia TEXT;
"""


def migrer_supabase(sb) -> None:
    """Ajoute les colonnes d'enrichissement si elles n'existent pas."""
    try:
        sb.rpc("exec_sql", {"query": NOUVELLES_COLONNES_SQL}).execute()
        print("[migration] Colonnes ajoutees via RPC exec_sql")
    except Exception:
        # exec_sql RPC n'est pas disponible sur tous les projets Supabase.
        # Dans ce cas, il faut executer le SQL manuellement dans l'editeur SQL.
        print("[migration] RPC exec_sql non disponible.")
        print("  → Executez ce SQL dans l'editeur Supabase :")
        print(NOUVELLES_COLONNES_SQL)


# ---------------------------------------------------------------------------
# Recuperation des annonces depuis Supabase
# ---------------------------------------------------------------------------

def charger_annonces_supabase(sb, limit: Optional[int] = None) -> list[dict]:
    """Charge les annonces non encore enrichies (sans score_marche)."""
    query = (
        sb.table("annonces")
        .select("id,titre,prix,surface,pieces,quartier,type_bien,lien")
        .is_("score_marche", "null")
        .order("id")
    )
    if limit:
        query = query.limit(limit)
    res = query.execute()
    return res.data or []


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def enrichir_tout(limit: Optional[int] = None) -> None:
    print("=" * 60)
    print("  ENRICHISSEMENT IA - Observatoire Immobilier Toulonnais")
    print("=" * 60)

    sb = _supabase()
    claude = _anthropic()

    # 1. Migration des colonnes
    print("\n[1/4] Migration des colonnes Supabase...")
    migrer_supabase(sb)

    # 2. Chargement de toutes les annonces pour calcul des medianes
    print("[2/4] Calcul des prix medians par quartier...")
    res_all = sb.table("annonces").select("quartier,prix,surface").execute()
    medianes = calculer_prix_medians(res_all.data or [])
    print(f"  → {len(medianes)} quartiers analyses")
    for q, m in sorted(medianes.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"     {q}: {m:.0f} €/m²")

    # 3. Chargement des annonces a enrichir (avec les bons noms de colonnes Supabase)
    print("[3/4] Chargement des annonces a enrichir...")
    annonces = charger_annonces_supabase(sb, limit=limit)
    print(f"  → {len(annonces)} annonces a traiter")

    if not annonces:
        print("  Rien a enrichir.")
        return

    # 4. Enrichissement
    print("[4/4] Enrichissement via Claude Haiku...\n")
    ok = 0
    erreurs = 0
    debut = time.time()

    for i, annonce in enumerate(annonces, 1):
        prix_m2 = float(annonce.get("prix_m2") or 0)
        quartier = annonce.get("quartier") or "Inconnu"
        median_q = medianes.get(quartier, 0)
        score_stat = score_marche_stat(prix_m2, median_q)

        try:
            champs = enrichir_annonce_claude(claude, annonce, score_stat)

            sb.table("annonces").update(champs).eq("id", annonce["id"]).execute()
            ok += 1

            tags_affich = champs.get("tags", "[]")
            print(
                f"  [{i:>4}/{len(annonces)}] {annonce['titre'][:45]:<45} "
                f"| {champs['score_marche']:<12} | couple:{champs['score_jeune_couple']} "
                f"| tags:{tags_affich}"
            )

        except anthropic.RateLimitError:
            print(f"  [{i}] Rate limit — pause 30s...")
            time.sleep(30)
            erreurs += 1
        except Exception as e:
            print(f"  [{i}] Erreur: {e}")
            erreurs += 1

        # Pause polie entre chaque appel (eviter rate limit)
        time.sleep(0.5)

    duree = time.time() - debut
    print(f"\n{'='*60}")
    print(f"  Termine en {duree:.0f}s")
    print(f"  Enrichies : {ok} / {len(annonces)}")
    print(f"  Erreurs   : {erreurs}")
    print(f"{'='*60}")

    # Mise a jour du CSV avec les donnees enrichies depuis Supabase
    if ok > 0:
        _mettre_a_jour_csv(sb)


CSV_PATH  = os.path.join(os.path.dirname(__file__), "..", "data", "annonces.csv")
JSON_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "annonces.json")

CSV_COLUMNS = [
    "titre", "prix", "surface", "pieces",
    "quartier", "type_bien", "prix_m2", "url", "source",
    "score_marche", "etage", "parking", "balcon", "vue_mer",
    "etat_bien", "score_jeune_couple", "tags", "resume_ia",
]

SELECT_FIELDS = (
    "titre,prix,surface,pieces,quartier,type_bien,lien,source,"
    "score_marche,etage,parking,balcon,vue_mer,"
    "etat_bien,score_jeune_couple,tags,resume_ia"
)


def _charger_toutes_annonces(sb) -> list[dict]:
    """Charge toutes les annonces depuis Supabase avec pagination."""
    toutes = []
    offset = 0
    while True:
        batch = (
            sb.table("annonces")
            .select(SELECT_FIELDS)
            .order("id")
            .range(offset, offset + 999)
            .execute()
            .data or []
        )
        toutes.extend(batch)
        if len(batch) < 1000:
            break
        offset += 1000
    return toutes


def _normaliser_row(row: dict) -> dict:
    """Normalise un enregistrement Supabase pour CSV/JSON."""
    row["url"] = row.pop("lien", "")
    if not row.get("prix_m2"):
        try:
            row["prix_m2"] = round(float(row["prix"]) / float(row["surface"]), 2)
        except (TypeError, ValueError, ZeroDivisionError):
            row["prix_m2"] = None
    # tags : deserialise si c'est une chaine JSON
    if isinstance(row.get("tags"), str):
        try:
            row["tags"] = json.loads(row["tags"])
        except (json.JSONDecodeError, TypeError):
            row["tags"] = []
    return row


def _mettre_a_jour_csv(sb) -> None:
    """Reecrit annonces.csv avec toutes les donnees depuis Supabase."""
    print("\n[Fichiers] Mise a jour CSV + JSON...")
    lignes = _charger_toutes_annonces(sb)
    annonces = [_normaliser_row(dict(r)) for r in lignes]

    # --- CSV ---
    chemin_csv = os.path.abspath(CSV_PATH)
    with open(chemin_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in annonces:
            # tags → JSON string pour le CSV
            row_csv = dict(row)
            if isinstance(row_csv.get("tags"), list):
                row_csv["tags"] = json.dumps(row_csv["tags"], ensure_ascii=False)
            writer.writerow(row_csv)
    print(f"  [CSV]  {len(annonces)} annonces → {chemin_csv}")

    # --- JSON ---
    chemin_json = os.path.abspath(JSON_PATH)
    with open(chemin_json, "w", encoding="utf-8") as f:
        json.dump(annonces, f, ensure_ascii=False, indent=2, default=str)
    print(f"  [JSON] {len(annonces)} annonces → {chemin_json}")


def rapport_enrichissement() -> None:
    """Affiche un rapport sur les donnees enrichies dans Supabase."""
    sb = _supabase()

    res = sb.table("annonces").select(
        "score_marche,score_jeune_couple,etat_bien,vue_mer,parking,balcon"
    ).not_.is_("score_marche", "null").execute()

    data = res.data or []
    if not data:
        print("Aucune annonce enrichie.")
        return

    print(f"\n=== Rapport enrichissement ({len(data)} annonces) ===\n")

    # Distribution scoring marche
    scores = {}
    for a in data:
        s = a.get("score_marche") or "?"
        scores[s] = scores.get(s, 0) + 1
    print("Scoring marche :")
    for s, n in sorted(scores.items(), key=lambda x: -x[1]):
        print(f"  {s:<16}: {n:>4} ({100*n/len(data):.1f}%)")

    # Score jeune couple moyen
    scores_couple = [a["score_jeune_couple"] for a in data if a.get("score_jeune_couple")]
    if scores_couple:
        moy = sum(scores_couple) / len(scores_couple)
        print(f"\nScore jeune couple moyen : {moy:.2f}/5")

    # Equipements
    avec_parking = sum(1 for a in data if a.get("parking"))
    avec_balcon = sum(1 for a in data if a.get("balcon"))
    vue_mer = sum(1 for a in data if a.get("vue_mer"))
    print(f"\nEquipements detectes :")
    print(f"  Parking : {avec_parking} ({100*avec_parking/len(data):.1f}%)")
    print(f"  Balcon  : {avec_balcon} ({100*avec_balcon/len(data):.1f}%)")
    print(f"  Vue mer : {vue_mer} ({100*vue_mer/len(data):.1f}%)")


# ---------------------------------------------------------------------------
# Point d'entree
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Enrichissement IA des annonces")
    parser.add_argument("--limit", type=int, default=None, help="Limiter le nombre d'annonces")
    parser.add_argument("--rapport", action="store_true", help="Afficher le rapport uniquement")
    args = parser.parse_args()

    if args.rapport:
        rapport_enrichissement()
    else:
        enrichir_tout(limit=args.limit)
