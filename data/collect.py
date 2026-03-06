"""
collect.py — Point d'entrée principal pour la collecte de données
Lance le pipeline DVF et prépare les données pour l'application.
"""

import os
import sys

# Ajouter le répertoire parent au path pour les imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dvf import run_pipeline, load_csv

# Chemin du CSV nettoyé (utilisé par toute l'app)
CLEAN_CSV_PATH = os.path.join(os.path.dirname(__file__), "raw", "dvf_clean.csv")


def get_data(force_refresh: bool = False) -> list[dict]:
    """
    Retourne les données DVF nettoyées.
    - Si le CSV existe et force_refresh=False : charge depuis le CSV (rapide)
    - Sinon : relance le pipeline complet (téléchargement + nettoyage)
    
    Usage dans Streamlit :
        from data.collect import get_data
        transactions = get_data()
    """
    if not force_refresh and os.path.exists(CLEAN_CSV_PATH):
        print(f"  📂 Chargement depuis cache : {CLEAN_CSV_PATH}")
        return load_csv(CLEAN_CSV_PATH)
    else:
        print("  🔄 Lancement du pipeline DVF...")
        return run_pipeline(output_csv=CLEAN_CSV_PATH)


if __name__ == "__main__":
    # Lance une collecte fraîche si appelé directement
    import argparse
    parser = argparse.ArgumentParser(description="Collecte données DVF Toulon")
    parser.add_argument("--refresh", action="store_true",
                        help="Force le re-téléchargement même si le cache existe")
    args = parser.parse_args()

    data = get_data(force_refresh=args.refresh)
    print(f"\n  ✅ {len(data)} transactions disponibles pour l'application.")
