"""
tests/test_dvf.py — Tests unitaires pour le nettoyage DVF
Lancés automatiquement par GitHub Actions (CI).
"""

import sys
import os
import csv
import tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dvf import clean_row, parse_float, parse_int, code_postal_to_quartier, clean_all, save_csv, load_csv, quick_stats


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


# ─────────────────────────────────────────
# Tests clean_all
# ─────────────────────────────────────────

def test_clean_all_retourne_valides():
    rows = [make_row(), make_row(type_local="Maison", surface_reelle_bati="100", valeur_fonciere="300000")]
    result = clean_all(rows)
    assert len(result) == 2

def test_clean_all_rejette_invalides():
    rows = [
        make_row(),
        make_row(type_local="Parking"),
        make_row(valeur_fonciere="1000"),
    ]
    result = clean_all(rows)
    assert len(result) == 1

def test_clean_all_liste_vide():
    assert clean_all([]) == []

def test_clean_all_tous_invalides():
    rows = [make_row(type_local="Terrain"), make_row(valeur_fonciere="999999999")]
    assert clean_all(rows) == []


# ─────────────────────────────────────────
# Tests save_csv / load_csv
# ─────────────────────────────────────────

def _make_clean_row():
    return {
        "date": "2024-03-15",
        "annee": 2024,
        "mois": 3,
        "type_bien": "Appartement",
        "surface": 65.0,
        "prix": 200000.0,
        "prix_m2": 3076.92,
        "pieces": 3,
        "code_postal": "83000",
        "quartier": "Centre-ville",
        "latitude": 43.12,
        "longitude": 5.93,
    }

def test_save_csv_cree_fichier():
    row = _make_clean_row()
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        path = f.name
    try:
        save_csv([row], path)
        assert os.path.exists(path)
    finally:
        os.unlink(path)

def test_save_csv_nombre_lignes():
    rows = [_make_clean_row(), _make_clean_row()]
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        path = f.name
    try:
        save_csv(rows, path)
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            lines = list(reader)
        # 1 ligne header + 2 lignes de données
        assert len(lines) == 3
    finally:
        os.unlink(path)

def test_load_csv_roundtrip():
    row = _make_clean_row()
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        path = f.name
    try:
        save_csv([row], path)
        loaded = load_csv(path)
        assert len(loaded) == 1
        assert loaded[0]["type_bien"] == "Appartement"
        assert loaded[0]["prix"] == 200000.0
        assert loaded[0]["surface"] == 65.0
        assert loaded[0]["annee"] == 2024
        assert loaded[0]["mois"] == 3
        assert loaded[0]["pieces"] == 3
    finally:
        os.unlink(path)

def test_load_csv_pieces_none():
    row = _make_clean_row()
    row["pieces"] = None
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        path = f.name
    try:
        save_csv([row], path)
        loaded = load_csv(path)
        assert loaded[0]["pieces"] is None
    finally:
        os.unlink(path)


# ─────────────────────────────────────────
# Tests quick_stats
# ─────────────────────────────────────────

def test_quick_stats_ne_plante_pas():
    rows = [_make_clean_row(), _make_clean_row()]
    quick_stats(rows)  # ne doit pas lever d'exception

def test_quick_stats_liste_vide(capsys):
    quick_stats([])
    captured = capsys.readouterr()
    assert "Aucune" in captured.out
