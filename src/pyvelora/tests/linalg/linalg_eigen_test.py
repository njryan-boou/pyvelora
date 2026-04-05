"""Tests for pyvelora/linalg/eigen.py."""
import numpy as np
import pytest
from pyvelora.core import Matrix, Vector
from pyvelora.linalg.eigen import eigenvalues, eigenvectors

A = Matrix(np.array([[2.0, 0.0], [0.0, 3.0]]))  # diagonal → trivial eigenvalues


def test_eigenvalues_returns_vector():
    vals = eigenvalues(A)
    assert isinstance(vals, Vector)


def test_eigenvalues_diagonal_matrix():
    vals = eigenvalues(A)
    assert set(np.round(vals.data.real, 8)) == {2.0, 3.0}


def test_eigenvectors_returns_matrix():
    V = eigenvectors(A)
    assert isinstance(V, Matrix)


def test_eigenvectors_shape():
    V = eigenvectors(A)
    assert V.shape == (2, 2)


def test_eigenvalues_non_square_raises():
    B = Matrix(np.ones((2, 3)))
    with pytest.raises(ValueError):
        eigenvalues(B)


def test_eigenvectors_non_square_raises():
    B = Matrix(np.ones((2, 3)))
    with pytest.raises(ValueError):
        eigenvectors(B)
