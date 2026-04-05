"""Tests for pyvelora/linalg/rref.py."""
import numpy as np
import pytest
from pyvelora.core import Matrix
from pyvelora.linalg.rref import rref


def test_rref_identity_unchanged():
    A = Matrix(np.eye(3))
    result, pivot_cols = rref(A)
    assert isinstance(result, Matrix)
    assert np.allclose(result.data, np.eye(3), atol=1e-10)
    assert len(pivot_cols) == 3
    assert pivot_cols == (0, 1, 2)


def test_rref_2x2():
    A = Matrix(np.array([[2.0, 4.0], [1.0, 3.0]]))
    result, pivot_cols = rref(A)
    assert np.allclose(result.data, np.eye(2), atol=1e-8)
    assert len(pivot_cols) == 2
    assert pivot_cols == (0, 1)


def test_rref_full_rank():
    A = Matrix(np.array([[1.0, 2.0, 3.0], [0.0, 1.0, 4.0], [0.0, 0.0, 1.0]]))
    result, pivot_cols = rref(A)
    assert np.allclose(result.data, np.eye(3), atol=1e-8)
    assert len(pivot_cols) == 3
    assert pivot_cols == (0, 1, 2)


def test_rref_rank_deficient():
    # Rows are linearly dependent
    A = Matrix(np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]]))
    result, pivot_cols = rref(A)
    # First row should be [1, 2, 3], second row should be [0, 0, 0]
    assert np.allclose(result.data[0], [1.0, 2.0, 3.0], atol=1e-8)
    assert np.allclose(result.data[1], [0.0, 0.0, 0.0], atol=1e-8)
    assert len(pivot_cols) == 1
    assert pivot_cols == (0,)


def test_rref_non_matrix_raises():
    with pytest.raises(TypeError):
        rref(np.eye(2))


def test_rref_does_not_modify_original():
    A = Matrix(np.array([[2.0, 4.0], [1.0, 3.0]]))
    original = A.data.copy()
    rref(A)
    assert np.allclose(A.data, original)
