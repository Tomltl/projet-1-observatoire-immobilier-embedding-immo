"""
Tests unitaires pour analysis/regression.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.regression import predict, error, sum_of_sqerrors, least_squares_fit, r_squared


def test_predict_basic():
    """predict() retourne alpha + beta * x."""
    assert predict(1.0, 2.0, 3.0) == 7.0
    assert predict(0.0, 5.0, 4.0) == 20.0


def test_error_basic():
    """error() retourne la difference entre prediction et valeur reelle."""
    assert error(1.0, 2.0, 3.0, 7.0) == 0.0
    assert error(1.0, 2.0, 3.0, 5.0) == 2.0


def test_least_squares_simple():
    """least_squares_fit() trouve alpha=1, beta=2 pour y = 2x + 1."""
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    y = [3.0, 5.0, 7.0, 9.0, 11.0]
    alpha, beta = least_squares_fit(x, y)
    assert abs(beta - 2.0) < 0.01
    assert abs(alpha - 1.0) < 0.01


def test_r_squared_perfect():
    """r_squared() retourne 1.0 pour une relation lineaire parfaite."""
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    y = [3.0, 5.0, 7.0, 9.0, 11.0]
    alpha, beta = least_squares_fit(x, y)
    assert abs(r_squared(alpha, beta, x, y) - 1.0) < 0.01


def test_sum_of_sqerrors_perfect():
    """sum_of_sqerrors() retourne 0 pour un fit parfait."""
    x = [1.0, 2.0, 3.0]
    y = [3.0, 5.0, 7.0]
    alpha, beta = least_squares_fit(x, y)
    assert sum_of_sqerrors(alpha, beta, x, y) < 1e-10
