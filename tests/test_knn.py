"""
tests/test_knn.py — Tests unitaires pour knn/distance.py et knn/similar.py
"""

import sys
import os
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from knn.distance import calculer_stats, distance, POIDS_DEFAUT
from knn.similar import knn_similar


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _bien(surface=60.0, prix_m2=3000.0, pieces=3, type_bien="Appartement", quartier="Centre-ville", **kw):
    return {
        "surface":   surface,
        "prix_m2":   prix_m2,
        "pieces":    pieces,
        "type_bien": type_bien,
        "quartier":  quartier,
        "date":      kw.get("date", "2023-01-01"),
        "prix":      kw.get("prix", surface * prix_m2),
    }


CORPUS_SIMPLE = [
    _bien(surface=50,  prix_m2=2800, pieces=2, quartier="Centre-ville",  date="2023-01-01", prix=140000),
    _bien(surface=65,  prix_m2=3100, pieces=3, quartier="Centre-ville",  date="2023-02-01", prix=201500),
    _bien(surface=90,  prix_m2=3500, pieces=4, quartier="Centre-ville",  date="2023-03-01", prix=315000),
    _bien(surface=40,  prix_m2=2500, pieces=1, quartier="Ouest (Le Mourillon / Saint-Jean-du-Var)", date="2023-04-01", prix=100000),
    _bien(surface=75,  prix_m2=3200, pieces=3, quartier="Ouest (Le Mourillon / Saint-Jean-du-Var)", date="2023-05-01", prix=240000),
    _bien(surface=120, prix_m2=2200, pieces=5, type_bien="Maison", quartier="Centre-ville", date="2023-06-01", prix=264000),
]


# ---------------------------------------------------------------------------
# calculer_stats
# ---------------------------------------------------------------------------

def test_calculer_stats_retourne_trois_cles():
    stats = calculer_stats(CORPUS_SIMPLE)
    assert "surface"  in stats
    assert "prix_m2"  in stats
    assert "pieces"   in stats


def test_calculer_stats_std_positive():
    stats = calculer_stats(CORPUS_SIMPLE)
    for feat, (mean, std) in stats.items():
        assert std > 0, f"std de '{feat}' doit être > 0"


def test_calculer_stats_corpus_uniforme_std_vaut_1():
    corpus = [_bien(surface=60) for _ in range(5)]
    stats = calculer_stats(corpus)
    _, std = stats["surface"]
    assert std == 1.0   # évite division par zéro


def test_calculer_stats_corpus_vide():
    stats = calculer_stats([])
    for feat in ("surface", "prix_m2", "pieces"):
        assert stats[feat] == (0.0, 1.0)


# ---------------------------------------------------------------------------
# distance
# ---------------------------------------------------------------------------

def test_distance_identique_vaut_zero():
    b = _bien()
    stats = calculer_stats([b])
    assert distance(b, b, stats) == 0.0


def test_distance_symetrique():
    a, b = CORPUS_SIMPLE[0], CORPUS_SIMPLE[2]
    stats = calculer_stats(CORPUS_SIMPLE)
    assert distance(a, b, stats) == distance(b, a, stats)


def test_distance_toujours_positive():
    stats = calculer_stats(CORPUS_SIMPLE)
    for i, a in enumerate(CORPUS_SIMPLE):
        for j, b in enumerate(CORPUS_SIMPLE):
            if i != j:
                assert distance(a, b, stats) >= 0.0


def test_distance_type_bien_different_penalise():
    """Deux biens identiques sauf type_bien → distance > 0."""
    a = _bien(type_bien="Appartement")
    b = _bien(type_bien="Maison")
    stats = calculer_stats([a, b])
    assert distance(a, b, stats) > 0.0


def test_distance_meme_quartier_plus_proche():
    """Bien dans le même quartier doit être plus proche qu'un bien hors quartier."""
    cible   = _bien(quartier="Centre-ville")
    proche  = _bien(surface=62, prix_m2=3050, pieces=3, quartier="Centre-ville")
    lointain = _bien(surface=62, prix_m2=3050, pieces=3, quartier="Est (La Valette / La Garde)")
    stats = calculer_stats([cible, proche, lointain])
    assert distance(cible, proche, stats) < distance(cible, lointain, stats)


def test_distance_feature_manquante_penalite_moderee():
    """Feature manquante (None) → pénalité, mais distance < distance maximale."""
    a = _bien()
    b = {**_bien(), "surface": None}
    stats = calculer_stats(CORPUS_SIMPLE)
    d = distance(a, b, stats)
    assert 0.0 < d < 100.0   # pénalité existe mais reste raisonnable


def test_distance_poids_custom():
    """Poids modifiés → distances différentes."""
    a, b = CORPUS_SIMPLE[0], CORPUS_SIMPLE[2]
    stats = calculer_stats(CORPUS_SIMPLE)
    d_def    = distance(a, b, stats, poids=POIDS_DEFAUT)
    d_custom = distance(a, b, stats, poids={**POIDS_DEFAUT, "prix_m2": 10.0})
    assert d_def != d_custom


# ---------------------------------------------------------------------------
# knn_similar
# ---------------------------------------------------------------------------

def test_knn_retourne_k_voisins():
    cible = _bien(surface=60, prix_m2=3000, pieces=3)
    voisins = knn_similar(cible, CORPUS_SIMPLE, k=3)
    assert len(voisins) == 3


def test_knn_trie_par_distance_croissante():
    cible = _bien(surface=60, prix_m2=3000, pieces=3)
    voisins = knn_similar(cible, CORPUS_SIMPLE, k=5)
    distances = [v["_distance"] for v in voisins]
    assert distances == sorted(distances)


def test_knn_champ_distance_present():
    cible = _bien()
    voisins = knn_similar(cible, CORPUS_SIMPLE, k=2)
    for v in voisins:
        assert "_distance" in v
        assert isinstance(v["_distance"], float)


def test_knn_exclure_cible_actif():
    """Quand la cible est dans le corpus, elle ne doit pas apparaître."""
    cible = CORPUS_SIMPLE[0]
    voisins = knn_similar(cible, CORPUS_SIMPLE, k=5, exclure_cible=True)
    empreintes = [(v["date"], v["prix"], v["surface"]) for v in voisins]
    assert (cible["date"], cible["prix"], cible["surface"]) not in empreintes


def test_knn_sans_exclure_cible():
    """Avec exclure_cible=False, le bien peut apparaître (distance 0)."""
    cible = CORPUS_SIMPLE[0]
    voisins = knn_similar(cible, CORPUS_SIMPLE, k=6, exclure_cible=False)
    distances = [v["_distance"] for v in voisins]
    assert 0.0 in distances


def test_knn_k_superieur_corpus():
    """k > len(corpus) → retourne tout le corpus (sans cible)."""
    cible = CORPUS_SIMPLE[0]
    voisins = knn_similar(cible, CORPUS_SIMPLE, k=100)
    assert len(voisins) == len(CORPUS_SIMPLE) - 1


def test_knn_corpus_vide_leve_erreur():
    import pytest
    cible = _bien()
    try:
        knn_similar(cible, [], k=5)
        assert False, "Devrait lever ValueError"
    except ValueError:
        pass


def test_knn_premier_voisin_est_le_plus_similaire():
    """Le voisin le plus proche doit être le plus similaire numériquement."""
    cible   = _bien(surface=60, prix_m2=3000, pieces=3, quartier="Centre-ville")
    proche  = _bien(surface=61, prix_m2=3010, pieces=3, quartier="Centre-ville", date="2023-07-01", prix=183610)
    lointain = _bien(surface=120, prix_m2=4500, pieces=6, quartier="Est (La Valette / La Garde)", date="2023-08-01", prix=540000)
    corpus  = [proche, lointain]
    voisins = knn_similar(cible, corpus, k=2)
    assert voisins[0]["surface"] == 61.0
