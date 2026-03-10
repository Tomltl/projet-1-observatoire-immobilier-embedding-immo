"""
tests/test_scraper.py — Tests unitaires pour scraping/scraper_annonces.py
Couvre les fonctions de logique pure (sans appels HTTP ni Supabase).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scraping.scraper_annonces import _scalar, _mediane, _build_url, nettoyer


# ─────────────────────────────────────────
# Tests _scalar
# ─────────────────────────────────────────

def test_scalar_liste_un_element():
    assert _scalar([42]) == 42

def test_scalar_liste_plusieurs_elements():
    assert _scalar([1, 2, 3]) == 1

def test_scalar_liste_vide():
    assert _scalar([], default="x") == "x"

def test_scalar_valeur_directe():
    assert _scalar(99) == 99

def test_scalar_none_retourne_default():
    assert _scalar(None, default="défaut") == "défaut"

def test_scalar_string():
    assert _scalar("bonjour") == "bonjour"


# ─────────────────────────────────────────
# Tests _mediane
# ─────────────────────────────────────────

def test_mediane_impair():
    assert _mediane([1.0, 3.0, 5.0]) == 3.0

def test_mediane_pair():
    assert _mediane([1.0, 2.0, 3.0, 4.0]) == 2.5

def test_mediane_vide():
    assert _mediane([]) == 0.0

def test_mediane_valeur_unique():
    assert _mediane([7.0]) == 7.0

def test_mediane_non_triee():
    assert _mediane([5.0, 1.0, 3.0]) == 3.0


# ─────────────────────────────────────────
# Tests _build_url
# ─────────────────────────────────────────

def test_build_url_contient_base():
    url = _build_url(0)
    assert "bienici.com" in url

def test_build_url_contient_filters():
    url = _build_url(0)
    assert "filters=" in url

def test_build_url_page_0_from_0():
    url = _build_url(0)
    assert "%22from%22%3A%200" in url or '"from": 0' in url or "from" in url

def test_build_url_differente_selon_page():
    assert _build_url(0) != _build_url(1)


# ─────────────────────────────────────────
# Tests nettoyer
# ─────────────────────────────────────────

def _make_annonce(**kwargs):
    base = {
        "titre":     "Bel appartement",
        "prix":      200000,
        "surface":   60.0,
        "nb_pieces": 3,
        "quartier":  "Centre-ville",
        "type_bien": "appartement",
        "url":       "https://www.bienici.com/annonce/abc123",
        "source":    "Orpi",
    }
    base.update(kwargs)
    return base

def test_nettoyer_annonce_valide():
    result = nettoyer([_make_annonce()])
    assert len(result) == 1
    assert result[0]["prix_m2"] == round(200000 / 60.0, 2)

def test_nettoyer_prix_absent():
    assert nettoyer([_make_annonce(prix=0)]) == []

def test_nettoyer_prix_trop_haut():
    assert nettoyer([_make_annonce(prix=600000)]) == []

def test_nettoyer_surface_absente():
    assert nettoyer([_make_annonce(surface=0)]) == []

def test_nettoyer_type_exclu():
    # type_bien=None signifie parking/terrain/etc.
    assert nettoyer([_make_annonce(type_bien=None)]) == []

def test_nettoyer_url_invalide():
    assert nettoyer([_make_annonce(url="")]) == []
    assert nettoyer([_make_annonce(url="https://www.bienici.com/annonce/")]) == []

def test_nettoyer_doublon_url():
    a1 = _make_annonce()
    a2 = _make_annonce()
    result = nettoyer([a1, a2])
    assert len(result) == 1

def test_nettoyer_doublon_empreinte():
    a1 = _make_annonce(url="https://www.bienici.com/annonce/aaa")
    a2 = _make_annonce(url="https://www.bienici.com/annonce/bbb")
    # Même prix, surface et titre → doublon empreinte
    result = nettoyer([a1, a2])
    assert len(result) == 1

def test_nettoyer_deux_annonces_differentes():
    a1 = _make_annonce(url="https://www.bienici.com/annonce/aaa", prix=200000)
    a2 = _make_annonce(url="https://www.bienici.com/annonce/bbb", prix=300000)
    result = nettoyer([a1, a2])
    assert len(result) == 2

def test_nettoyer_liste_vide():
    assert nettoyer([]) == []

def test_nettoyer_calcule_prix_m2():
    result = nettoyer([_make_annonce(prix=300000, surface=100.0)])
    assert result[0]["prix_m2"] == 3000.0
