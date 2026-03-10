"""
tests/test_stats.py — Tests unitaires pour analysis/stats.py
"""

import sys
import os
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.stats import mean, median, variance, standard_deviation, covariance, correlation


# ─────────────────────────────────────────
# Tests mean
# ─────────────────────────────────────────

def test_mean_simple():
    assert mean([1, 2, 3, 4, 5]) == 3.0

def test_mean_deux_valeurs():
    assert mean([10, 20]) == 15.0

def test_mean_valeur_unique():
    assert mean([42.0]) == 42.0

def test_mean_nombres_negatifs():
    assert mean([-2, 0, 2]) == 0.0


# ─────────────────────────────────────────
# Tests median
# ─────────────────────────────────────────

def test_median_impair():
    assert median([1, 2, 3]) == 2

def test_median_pair():
    assert median([1, 2, 3, 4]) == 2.5

def test_median_non_trie():
    assert median([5, 1, 3]) == 3

def test_median_valeur_unique():
    assert median([7]) == 7


# ─────────────────────────────────────────
# Tests variance
# ─────────────────────────────────────────

def test_variance_connue():
    # variance([2,4,4,4,5,5,7,9]) = 4.0
    result = variance([2, 4, 4, 4, 5, 5, 7, 9])
    assert abs(result - 4.0) < 0.1

def test_variance_valeurs_identiques():
    assert variance([5, 5, 5, 5]) == 0.0

def test_variance_positive():
    assert variance([1, 2, 3, 4, 5]) > 0


# ─────────────────────────────────────────
# Tests standard_deviation
# ─────────────────────────────────────────

def test_std_connue():
    result = standard_deviation([2, 4, 4, 4, 5, 5, 7, 9])
    assert abs(result - 2.0) < 0.1

def test_std_valeurs_identiques():
    assert standard_deviation([3, 3, 3]) == 0.0

def test_std_est_racine_variance():
    xs = [1, 2, 3, 4, 5]
    assert abs(standard_deviation(xs) - math.sqrt(variance(xs))) < 1e-10


# ─────────────────────────────────────────
# Tests covariance
# ─────────────────────────────────────────

def test_covariance_positive():
    xs = [1, 2, 3, 4, 5]
    ys = [2, 4, 6, 8, 10]
    assert covariance(xs, ys) > 0

def test_covariance_negative():
    xs = [1, 2, 3, 4, 5]
    ys = [10, 8, 6, 4, 2]
    assert covariance(xs, ys) < 0

def test_covariance_independants():
    # covariance d'une liste avec elle-même = variance
    xs = [1, 2, 3, 4, 5]
    assert abs(covariance(xs, xs) - variance(xs)) < 1e-10


# ─────────────────────────────────────────
# Tests correlation
# ─────────────────────────────────────────

def test_correlation_parfaite():
    xs = [1.0, 2.0, 3.0, 4.0, 5.0]
    result = correlation(xs, xs)
    assert abs(result - 1.0) < 0.01

def test_correlation_inverse():
    xs = [1.0, 2.0, 3.0, 4.0, 5.0]
    ys = [5.0, 4.0, 3.0, 2.0, 1.0]
    result = correlation(xs, ys)
    assert abs(result - (-1.0)) < 0.01

def test_correlation_bornee():
    xs = [1, 2, 3, 4, 5]
    ys = [2, 4, 5, 4, 5]
    result = correlation(xs, ys)
    assert -1.0 <= result <= 1.0

def test_correlation_std_nulle():
    # Si std = 0, retourne 0 sans lever d'exception
    xs = [3, 3, 3]
    ys = [1, 2, 3]
    assert correlation(xs, ys) == 0
