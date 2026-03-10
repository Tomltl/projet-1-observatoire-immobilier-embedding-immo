"""
tests/test_enrichissement.py — Tests unitaires pour scraping/enrichissement.py
Couvre les fonctions de logique pure (sans appels API ni Supabase).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scraping.enrichissement import _mediane, _prix_m2, calculer_prix_medians, score_marche_stat


# ─────────────────────────────────────────
# Tests _mediane
# ─────────────────────────────────────────

def test_mediane_impair():
    assert _mediane([1.0, 2.0, 3.0]) == 2.0

def test_mediane_pair():
    assert _mediane([1.0, 2.0, 3.0, 4.0]) == 2.5

def test_mediane_vide():
    assert _mediane([]) == 0.0

def test_mediane_filtre_zeros():
    # Les valeurs <= 0 doivent être ignorées
    assert _mediane([0.0, 0.0, 3.0]) == 3.0

def test_mediane_valeur_unique():
    assert _mediane([42.0]) == 42.0


# ─────────────────────────────────────────
# Tests _prix_m2
# ─────────────────────────────────────────

def test_prix_m2_normal():
    annonce = {"prix": 200000, "surface": 50}
    assert _prix_m2(annonce) == 4000.0

def test_prix_m2_surface_zero():
    annonce = {"prix": 200000, "surface": 0}
    assert _prix_m2(annonce) == 0.0

def test_prix_m2_surface_absente():
    annonce = {"prix": 200000}
    assert _prix_m2(annonce) == 0.0

def test_prix_m2_prix_absent():
    annonce = {"surface": 50}
    assert _prix_m2(annonce) == 0.0

def test_prix_m2_valeurs_string():
    annonce = {"prix": "150000", "surface": "60"}
    assert _prix_m2(annonce) == 2500.0


# ─────────────────────────────────────────
# Tests calculer_prix_medians
# ─────────────────────────────────────────

def test_calculer_prix_medians_basic():
    annonces = [
        {"quartier": "Centre-ville", "prix": 200000, "surface": 50},
        {"quartier": "Centre-ville", "prix": 300000, "surface": 60},
        {"quartier": "Mourillon",    "prix": 250000, "surface": 50},
    ]
    result = calculer_prix_medians(annonces)
    assert "Centre-ville" in result
    assert "Mourillon" in result
    assert result["Mourillon"] == 5000.0

def test_calculer_prix_medians_liste_vide():
    assert calculer_prix_medians([]) == {}

def test_calculer_prix_medians_quartier_inconnu():
    annonces = [{"prix": 100000, "surface": 50}]
    result = calculer_prix_medians(annonces)
    assert "Inconnu" in result

def test_calculer_prix_medians_ignore_prix_m2_nul():
    annonces = [
        {"quartier": "Centre", "prix": 0, "surface": 50},
        {"quartier": "Centre", "prix": 200000, "surface": 50},
    ]
    result = calculer_prix_medians(annonces)
    assert result["Centre"] == 4000.0


# ─────────────────────────────────────────
# Tests score_marche_stat
# ─────────────────────────────────────────

def test_score_opportunite():
    # prix_m2 = 2000, median = 3000 → ratio 0.67 < 0.90 → Opportunite
    assert score_marche_stat(2000.0, 3000.0) == "Opportunite"

def test_score_surevalue():
    # prix_m2 = 4000, median = 3000 → ratio 1.33 > 1.10 → Surevalue
    assert score_marche_stat(4000.0, 3000.0) == "Surevalue"

def test_score_prix_marche():
    # prix_m2 = 3000, median = 3000 → ratio 1.0 → Prix marche
    assert score_marche_stat(3000.0, 3000.0) == "Prix marche"

def test_score_limite_basse():
    # ratio exactement 0.90 → Prix marche (non < 0.90)
    assert score_marche_stat(2700.0, 3000.0) == "Prix marche"

def test_score_limite_haute():
    # ratio exactement 1.10 → Prix marche (non > 1.10)
    assert score_marche_stat(3300.0, 3000.0) == "Prix marche"

def test_score_median_zero():
    # Médiane inconnue → Prix marche par défaut
    assert score_marche_stat(3000.0, 0.0) == "Prix marche"

def test_score_median_none():
    assert score_marche_stat(3000.0, None) == "Prix marche"
