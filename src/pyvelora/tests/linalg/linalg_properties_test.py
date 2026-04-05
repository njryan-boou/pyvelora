"""Tests for pyvelora/linalg/properties.py."""
import numpy as np
import pytest
from pyvelora.core import Matrix
from pyvelora.linalg.properties import (
    trace, rank, minor, cofactor, cofactor_matrix, adjugate,
    determinant, inverse,
)

A = Matrix(np.array([[1.0, 2.0], [3.0, 4.0]]))
I3 = Matrix(np.eye(3))


def test_trace():
    assert np.isclose(trace(A), 5.0)


def test_trace_identity():
    assert np.isclose(trace(I3), 3.0)


def test_rank_full():
    assert rank(A) == 2


def test_rank_singular():
    B = Matrix(np.array([[1.0, 2.0], [2.0, 4.0]]))
    assert rank(B) == 1


def test_minor_value():
    # minor of A at (0,0) is det([[4]]) = 4
    assert np.isclose(minor(A, 0, 0), 4.0)


def test_cofactor_sign():
    # cofactor(A, 0, 1) = (-1)^(0+1) * minor(A, 0, 1) = -1 * 3 = -3
    assert np.isclose(cofactor(A, 0, 1), -3.0)


def test_cofactor_matrix_shape():
    C = cofactor_matrix(A)
    assert isinstance(C, Matrix)
    assert C.shape == (2, 2)


def test_adjugate():
    adj = adjugate(A)
    assert isinstance(adj, Matrix)
    # A * adj(A) should equal det(A) * I
    det = determinant(A)
    assert np.allclose((np.array(A.data) @ np.array(adj.data)), det * np.eye(2), atol=1e-8)


def test_determinant_2x2():
    # det([[1,2],[3,4]]) = 4 - 6 = -2
    assert np.isclose(determinant(A), -2.0)


def test_determinant_identity():
    assert np.isclose(determinant(I3), 1.0)


def test_inverse_reverses_matrix():
    inv = inverse(A)
    assert np.allclose(np.array(A.data) @ np.array(inv.data), np.eye(2), atol=1e-8)
