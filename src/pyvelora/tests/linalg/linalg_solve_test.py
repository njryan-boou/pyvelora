"""Tests for pyvelora/linalg/solve.py."""
import numpy as np
import pytest
from pyvelora.core import Vector, Matrix
from pyvelora.linalg.solve import (
    forward_substitution, backward_substitution, solve_linear_system,
)

L = Matrix(np.array([[1.0, 0.0], [2.0, 1.0]]))   # lower triangular
U = Matrix(np.array([[3.0, 1.0], [0.0, 2.0]]))   # upper triangular
b = Vector(np.array([3.0, 8.0]))


def test_forward_substitution_returns_vector():
    x = forward_substitution(L, b)
    assert isinstance(x, Vector)


def test_forward_substitution_solves():
    x = forward_substitution(L, b)
    assert np.allclose(np.array(L.data) @ np.array(x.data), np.array(b.data), atol=1e-8)


def test_backward_substitution_returns_vector():
    x = backward_substitution(U, b)
    assert isinstance(x, Vector)


def test_backward_substitution_solves():
    x = backward_substitution(U, b)
    assert np.allclose(np.array(U.data) @ np.array(x.data), np.array(b.data), atol=1e-8)


def test_solve_linear_system():
    A = Matrix(np.array([[2.0, 1.0], [5.0, 7.0]]))
    rhs = Vector(np.array([11.0, 13.0]))
    x = solve_linear_system(A, rhs)
    assert isinstance(x, Vector)
    assert np.allclose(np.array(A.data) @ np.array(x.data), np.array(rhs.data), atol=1e-8)


def test_forward_substitution_non_square_raises():
    bad = Matrix(np.ones((2, 3)))
    with pytest.raises(ValueError):
        forward_substitution(bad, Vector(np.ones(2)))


def test_backward_substitution_dimension_mismatch_raises():
    with pytest.raises(ValueError):
        backward_substitution(U, Vector(np.ones(3)))


def test_solve_linear_system_non_square_raises():
    A = Matrix(np.ones((2, 3)))
    with pytest.raises(ValueError):
        solve_linear_system(A, Vector(np.ones(2)))
