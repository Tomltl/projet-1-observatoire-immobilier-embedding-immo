"""
knn/similar.py — Recherche des k biens les plus similaires (k-NN).

Usage :
    from knn.similar import knn_similar

    voisins = knn_similar(cible=bien, corpus=transactions, k=5)
    # → liste de 5 dicts triés par distance croissante
    # → chaque dict contient "_distance" (float)
"""

from __future__ import annotations

from knn.distance import calculer_stats, distance, POIDS_DEFAUT


# ---------------------------------------------------------------------------
# Identité fonctionnelle
# ---------------------------------------------------------------------------

def _empreinte(bien: dict) -> tuple:
    """
    Clé d'identité d'un bien pour l'auto-exclusion.

    On utilise (date, prix, surface) plutôt qu'un id car les deux sources
    (DVF et Bienici) n'ont pas de champ id commun.
    """
    return (
        bien.get("date", ""),
        bien.get("prix", 0),
        bien.get("surface", 0.0),
    )


# ---------------------------------------------------------------------------
# k-NN
# ---------------------------------------------------------------------------

def knn_similar(
    cible: dict,
    corpus: list[dict],
    k: int = 5,
    poids: dict[str, float] | None = None,
    exclure_cible: bool = True,
) -> list[dict]:
    """
    Retourne les k biens les plus proches de `cible` dans `corpus`.

    Args:
        cible:          Le bien de référence (doit avoir surface, prix_m2,
                        pieces, type_bien, quartier).
        corpus:         Liste de tous les biens candidats.
        k:              Nombre de voisins à retourner (défaut : 5).
        poids:          Poids des features. None → POIDS_DEFAUT.
        exclure_cible:  Si True, exclut le bien lui-même du résultat
                        (utile quand `cible` est dans `corpus`).

    Returns:
        Liste de dicts triée par distance croissante.
        Chaque dict contient le champ supplémentaire "_distance" (float).

    Raises:
        ValueError: si corpus est vide.
    """
    if not corpus:
        raise ValueError("Le corpus est vide — impossible de calculer les voisins.")

    if poids is None:
        poids = POIDS_DEFAUT

    # Normalisation calculée une seule fois sur tout le corpus
    stats = calculer_stats(corpus)

    empreinte_cible = _empreinte(cible) if exclure_cible else None

    candidats: list[tuple[float, dict]] = []

    for bien in corpus:
        if exclure_cible and _empreinte(bien) == empreinte_cible:
            continue

        d = distance(cible, bien, stats, poids)
        candidats.append((d, bien))

    candidats.sort(key=lambda x: x[0])

    return [
        {**bien, "_distance": round(d, 4)}
        for d, bien in candidats[:k]
    ]
