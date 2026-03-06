"""
tests/test_dvf.py — Tests unitaires pour le nettoyage DVF
Lancés automatiquement par GitHub Actions (CI).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dvf import clean_row, parse_float, parse_int, code_postal_to_quartier


# ─────────────────────────────────────────
# Tests parse_float
# ─────────────────────────────────────────

def test_parse_float_normal():
    assert parse_float("250000") == 250000.0

def test_parse_float_virgule():
    assert parse_float("3,5") == 3.5

def test_parse_float_none():
    assert parse_float(None) is None

def test_parse_float_vide():
    assert parse_float("") is None

def test_parse_float_invalide():
    assert parse_float("abc") is None


# ─────────────────────────────────────────
# Tests parse_int
# ─────────────────────────────────────────

def test_parse_int_normal():
    assert parse_int("3") == 3

def test_parse_int_float_str():
    assert parse_int("3.0") == 3

def test_parse_int_none():
    assert parse_int(None) is None


# ─────────────────────────────────────────
# Tests code_postal_to_quartier
# ─────────────────────────────────────────

def test_quartier_centre():
    assert code_postal_to_quartier("83000") == "Centre-ville"

def test_quartier_ouest():
    assert "Ouest" in code_postal_to_quartier("83100")

def test_quartier_est():
    assert "Est" in code_postal_to_quartier("83200")

def test_quartier_inconnu():
    result = code_postal_to_quartier("99999")
    assert "Autre" in result


# ─────────────────────────────────────────
# Tests clean_row — cas valides
# ─────────────────────────────────────────

def make_row(**kwargs):
    """Crée une ligne DVF valide par défaut, modifiable par kwargs."""
    base = {
        "type_local": "Appartement",
        "valeur_fonciere": "200000",
        "surface_reelle_bati": "65",
        "nombre_pieces_principales": "3",
        "date_mutation": "2024-03-15",
        "code_postal": "83000",
    }
    base.update(kwargs)
    return base


def test_clean_row_valide():
    row = clean_row(make_row())
    assert row is not None
    assert row["type_bien"] == "Appartement"
    assert row["prix"] == 200000.0
    assert row["surface"] == 65.0
    assert abs(row["prix_m2"] - 3076.92) < 1
    assert row["quartier"] == "Centre-ville"
    assert row["annee"] == 2024
    assert row["mois"] == 3


def test_clean_row_maison():
    row = clean_row(make_row(type_local="Maison", surface_reelle_bati="100", valeur_fonciere="350000"))
    assert row is not None
    assert row["type_bien"] == "Maison"


def test_clean_row_prix_m2_calcule():
    row = clean_row(make_row(valeur_fonciere="300000", surface_reelle_bati="60"))
    assert row is not None
    assert row["prix_m2"] == 5000.0


# ─────────────────────────────────────────
# Tests clean_row — cas rejetés
# ─────────────────────────────────────────

def test_clean_row_type_invalide():
    assert clean_row(make_row(type_local="Local commercial")) is None

def test_clean_row_prix_trop_bas():
    assert clean_row(make_row(valeur_fonciere="1000")) is None

def test_clean_row_prix_trop_haut():
    assert clean_row(make_row(valeur_fonciere="5000000")) is None

def test_clean_row_surface_trop_petite():
    assert clean_row(make_row(surface_reelle_bati="3")) is None

def test_clean_row_surface_trop_grande():
    assert clean_row(make_row(surface_reelle_bati="600")) is None

def test_clean_row_prix_manquant():
    assert clean_row(make_row(valeur_fonciere=None)) is None

def test_clean_row_surface_manquante():
    assert clean_row(make_row(surface_reelle_bati="")) is None

def test_clean_row_date_invalide():
    assert clean_row(make_row(date_mutation="pas-une-date")) is None

def test_clean_row_prix_m2_aberrant_bas():
    # Surface énorme pour un prix bas → prix/m² < 500
    assert clean_row(make_row(valeur_fonciere="40000", surface_reelle_bati="500")) is None

def test_clean_row_prix_m2_aberrant_haut():
    # 450k pour 9m² → prix/m² > 15000
    assert clean_row(make_row(valeur_fonciere="450000", surface_reelle_bati="9")) is None
