"""
knn/ — Module de recherche de biens similaires (k-NN).

API publique :
    calculer_stats(corpus)              → stats de normalisation
    distance(a, b, stats, poids)        → float
    knn_similar(cible, corpus, k, ...)  → list[dict]
    POIDS_DEFAUT                        → dict[str, float]
"""

from knn.distance import distance, calculer_stats, POIDS_DEFAUT
from knn.similar import knn_similar

__all__ = ["distance", "calculer_stats", "knn_similar", "POIDS_DEFAUT"]
