"""Tests for pyvelora/linalg/checks.py."""
import numpy as np
import pytest
from pyvelora.core import Matrix
from pyvelora.linalg.checks import (
    is_square, is_symmetric, is_orthogonal, is_singular,
    is_invertible, is_diagonal, is_identity,
    is_upper_triangular, is_lower_triangular, is_rref,
    is_skew_symmetric, is_positive_definite,
)


def test_is_square_true():
    assert is_square(Matrix(np.eye(3))) is True


def test_is_square_false():
    assert is_square(Matrix(np.ones((2, 3)))) is False


def test_is_symmetric_true():
    A = Matrix(np.array([[1.0, 2.0], [2.0, 3.0]]))
    assert is_symmetric(A) is True


def test_is_symmetric_false():
    A = Matrix(np.array([[1.0, 2.0], [3.0, 4.0]]))
    assert is_symmetric(A) is False


def test_is_orthogonal_true():
    Q = Matrix(np.array([[0.0, -1.0], [1.0, 0.0]]))
    assert is_orthogonal(Q) is True


def test_is_orthogonal_false():
    assert is_orthogonal(Matrix(np.array([[2.0, 0.0], [0.0, 2.0]]))) is False


def test_is_singular_true():
    A = Matrix(np.array([[1.0, 2.0], [2.0, 4.0]]))
    assert is_singular(A) is True


def test_is_invertible_true():
    assert is_invertible(Matrix(np.eye(3))) is True


def test_is_diagonal_true():
    assert is_diagonal(Matrix(np.diag([1.0, 2.0, 3.0]))) is True


def test_is_diagonal_false():
    assert is_diagonal(Matrix(np.array([[1.0, 1.0], [0.0, 1.0]]))) is False


def test_is_identity_true():
    assert is_identity(Matrix(np.eye(3))) is True


def test_is_identity_false():
    assert is_identity(Matrix(np.diag([1.0, 2.0]))) is False


def test_is_upper_triangular():
    U = Matrix(np.array([[1.0, 2.0], [0.0, 3.0]]))
    assert is_upper_triangular(U) is True


def test_is_lower_triangular():
    L = Matrix(np.array([[1.0, 0.0], [2.0, 3.0]]))
    assert is_lower_triangular(L) is True


def test_is_rref_identity():
    assert is_rref(Matrix(np.eye(2))) is True


def test_is_rref_non_rref():
    A = Matrix(np.array([[2.0, 4.0], [0.0, 1.0]]))
    assert is_rref(A) is False


def test_is_skew_symmetric_true():
    A = Matrix(np.array([[0.0, -2.0], [2.0, 0.0]]))
    assert is_skew_symmetric(A) is True


def test_is_skew_symmetric_false():
    A = Matrix(np.array([[0.0, 2.0], [2.0, 0.0]]))
    assert is_skew_symmetric(A) is False


def test_is_positive_definite_true():
    A = Matrix(np.array([[2.0, -1.0], [-1.0, 2.0]]))
    assert is_positive_definite(A) is True


def test_is_positive_definite_false_for_indefinite():
    A = Matrix(np.array([[1.0, 2.0], [2.0, 1.0]]))
    assert is_positive_definite(A) is False


def test_is_positive_definite_false_for_non_symmetric():
    A = Matrix(np.array([[1.0, 2.0], [0.0, 1.0]]))
    assert is_positive_definite(A) is False
