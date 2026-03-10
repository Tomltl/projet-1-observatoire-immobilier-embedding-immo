"""
knn/demo.py — Test rapide du module KNN sur les vraies données DVF.

Usage :
    python -m knn.demo
    python -m knn.demo --surface 65 --pieces 3 --quartier "Centre-ville"
    python -m knn.demo --k 10
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.collect import get_data
from knn.similar import knn_similar
from knn.distance import POIDS_DEFAUT


# ---------------------------------------------------------------------------
# Affichage
# ---------------------------------------------------------------------------

def _ligne(label: str, valeur: str, width: int = 22) -> str:
    return f"  {label:<{width}} {valeur}"


def afficher_bien(bien: dict, titre: str = "", index: int | None = None) -> None:
    prefix = f"[{index}] " if index is not None else ""
    print(f"\n  {prefix}{titre or bien.get('date', '—')}")
    print(_ligne("Type :",        bien.get("type_bien", "—")))
    print(_ligne("Quartier :",    bien.get("quartier",  "—")))
    print(_ligne("Surface :",     f"{bien.get('surface', 0):.0f} m²"))
    print(_ligne("Pièces :",      str(bien.get("pieces", "—"))))
    print(_ligne("Prix :",        f"{bien.get('prix', 0):,.0f} €"))
    print(_ligne("Prix/m² :",     f"{bien.get('prix_m2', 0):,.0f} €/m²"))
    if "_distance" in bien:
        print(_ligne("Distance KNN :", f"{bien['_distance']:.4f}"))


def afficher_separateur(char: str = "─", width: int = 52) -> None:
    print(char * width)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Démo KNN — biens similaires")
    parser.add_argument("--surface",  type=float, default=65.0,           help="Surface de la cible (m²)")
    parser.add_argument("--pieces",   type=int,   default=3,              help="Nombre de pièces de la cible")
    parser.add_argument("--quartier", type=str,   default="Centre-ville", help="Quartier de la cible")
    parser.add_argument("--type",     type=str,   default="Appartement",  help="Type de bien (Appartement / Maison)")
    parser.add_argument("--k",        type=int,   default=5,              help="Nombre de voisins à afficher")
    parser.add_argument("--index",    type=int,   default=None,           help="Utilise le bien n°INDEX du corpus comme cible")
    args = parser.parse_args()

    # ── Chargement des données ───────────────────────────────────────────────
    print("\nChargement des données DVF...")
    corpus = get_data()
    print(f"  {len(corpus)} transactions chargées.")

    # ── Cible ───────────────────────────────────────────────────────────────
    if args.index is not None:
        if args.index >= len(corpus):
            print(f"Erreur : index {args.index} hors limites (corpus = {len(corpus)} lignes).")
            sys.exit(1)
        cible = corpus[args.index]
        print(f"\n  Cible = transaction n°{args.index} du corpus (auto-exclue des voisins).")
    else:
        # Cible fictive construite depuis les arguments CLI
        prix_m2_estime = 3000.0   # valeur neutre si non renseignée
        cible = {
            "surface":   args.surface,
            "pieces":    args.pieces,
            "quartier":  args.quartier,
            "type_bien": args.type,
            "prix_m2":   prix_m2_estime,
            "prix":      args.surface * prix_m2_estime,
            "date":      "",
        }
        print("\n  Cible = bien fictif construit depuis les arguments CLI.")

    # ── KNN ─────────────────────────────────────────────────────────────────
    voisins = knn_similar(
        cible=cible,
        corpus=corpus,
        k=args.k,
        exclure_cible=(args.index is not None),
    )

    # ── Affichage ────────────────────────────────────────────────────────────
    afficher_separateur("═")
    print("  BIEN DE RÉFÉRENCE")
    afficher_separateur("═")
    afficher_bien(cible, titre="Cible")

    afficher_separateur()
    print(f"  {args.k} BIENS LES PLUS SIMILAIRES  (poids : {POIDS_DEFAUT})")
    afficher_separateur()

    for i, v in enumerate(voisins, 1):
        afficher_bien(v, titre=v.get("date", f"Bien {i}"), index=i)

    # ── Résumé distances ─────────────────────────────────────────────────────
    afficher_separateur()
    distances = [v["_distance"] for v in voisins]
    print(f"\n  Distance min  : {min(distances):.4f}")
    print(f"  Distance max  : {max(distances):.4f}")
    print(f"  Distance moy. : {sum(distances)/len(distances):.4f}")
    afficher_separateur("═")


if __name__ == "__main__":
    main()
