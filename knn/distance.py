"""
knn/distance.py — Distance entre deux biens immobiliers.

Formule : distance euclidienne pondérée sur features normalisées.

Features numériques (normalisées par z-score) :
    surface, prix_m2, pieces

Features catégorielles (pénalité binaire 0/1) :
    type_bien, quartier

Usage :
    stats = calculer_stats(corpus)
    d = distance(bien_a, bien_b, stats)
"""

from __future__ import annotations

import math

# ---------------------------------------------------------------------------
# Poids par défaut
# Ajustables par l'appelant selon le contexte (ex. : forcer le même quartier)
# ---------------------------------------------------------------------------

POIDS_DEFAUT: dict[str, float] = {
    "surface":   1.5,   # surface habitable — critère fort
    "prix_m2":   2.0,   # positionnement marché — critère le plus discriminant
    "pieces":    1.0,   # nombre de pièces — important mais secondaire
    "type_bien": 2.0,   # appartement vs maison — rarement interchangeable
    "quartier":  1.5,   # localisation — fort mais pas rédhibitoire
}

FEATURES_NUM: tuple[str, ...] = ("surface", "prix_m2", "pieces")
FEATURES_CAT: tuple[str, ...] = ("type_bien", "quartier")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_float(val) -> float | None:
    """Convertit une valeur en float, None si impossible."""
    if val is None or val == "":
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def calculer_stats(corpus: list[dict]) -> dict[str, tuple[float, float]]:
    """
    Calcule (moyenne, écart-type) pour chaque feature numérique sur le corpus.

    Ces stats servent à normaliser les features avant le calcul de distance,
    pour qu'une grande valeur absolue (ex. prix_m2 en milliers) n'écrase pas
    une petite valeur (ex. pieces entre 1 et 6).

    Returns:
        {"surface": (mean, std), "prix_m2": (mean, std), "pieces": (mean, std)}
        std vaut 1.0 si l'écart-type est quasi nul (corpus uniforme).
    """
    stats: dict[str, tuple[float, float]] = {}

    for feat in FEATURES_NUM:
        valeurs = [v for b in corpus if (v := _to_float(b.get(feat))) is not None]

        if not valeurs:
            stats[feat] = (0.0, 1.0)
            continue

        n    = len(valeurs)
        mean = sum(valeurs) / n
        var  = sum((v - mean) ** 2 for v in valeurs) / n
        std  = math.sqrt(var) if var > 1e-9 else 1.0
        stats[feat] = (mean, std)

    return stats


# ---------------------------------------------------------------------------
# Distance
# ---------------------------------------------------------------------------

def distance(
    a: dict,
    b: dict,
    stats: dict[str, tuple[float, float]],
    poids: dict[str, float] = POIDS_DEFAUT,
) -> float:
    """
    Distance euclidienne pondérée et normalisée entre deux biens.

    Pour chaque feature numérique f :
        terme = poids[f] * ((a[f] - b[f]) / std_f) ** 2

    Pour chaque feature catégorielle c :
        terme = poids[c] * (0 si a[c] == b[c] else 1)

    Feature manquante chez l'un des deux biens → pénalité fixe de 0.5 * poids[f].

    Returns:
        sqrt(somme des termes) — toujours >= 0.
    """
    total = 0.0

    # ── Features numériques ─────────────────────────────────────────────────
    for feat in FEATURES_NUM:
        w  = poids.get(feat, 1.0)
        va = _to_float(a.get(feat))
        vb = _to_float(b.get(feat))

        if va is None or vb is None:
            # Valeur manquante : pénalité fixe modérée
            total += w * 0.5
            continue

        _, std = stats.get(feat, (0.0, 1.0))
        diff   = (va - vb) / std
        total += w * diff * diff

    # ── Features catégorielles ───────────────────────────────────────────────
    for feat in FEATURES_CAT:
        w = poids.get(feat, 1.0)
        if a.get(feat) != b.get(feat):
            total += w * 1.0   # pénalité binaire pleine

    return math.sqrt(total)
