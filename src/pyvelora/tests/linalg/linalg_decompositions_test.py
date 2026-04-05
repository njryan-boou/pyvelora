"""Tests for pyvelora/linalg/decompositions.py."""
import numpy as np
import pytest
from pyvelora.core import Matrix, Vector
from pyvelora.linalg.decompositions import (
    lu_decomposition, qr_decomposition, svd_decomposition,
    eigen_decomposition, cholesky_decomposition,
    schur_decomposition, polar_decomposition,
)

A2 = Matrix(np.array([[4.0, 3.0], [6.0, 3.0]]))
SPD = Matrix(np.array([[4.0, 2.0], [2.0, 3.0]]))  # symmetric positive definite


def test_lu_returns_matrices():
    L, U = lu_decomposition(A2)
    assert isinstance(L, Matrix)
    assert isinstance(U, Matrix)


def test_lu_product_equals_original():
    L, U = lu_decomposition(A2)
    assert np.allclose(np.array(L.data) @ np.array(U.data), np.array(A2.data), atol=1e-8)


def test_qr_orthogonality():
    Q, R = qr_decomposition(A2)
    assert isinstance(Q, Matrix)
    q = np.array(Q.data)
    assert np.allclose(q.T @ q, np.eye(2), atol=1e-8)


def test_qr_product_equals_original():
    Q, R = qr_decomposition(A2)
    assert np.allclose(np.array(Q.data) @ np.array(R.data), np.array(A2.data), atol=1e-8)


def test_svd_returns_three_matrices():
    U, S, Vh = svd_decomposition(A2)
    assert isinstance(U, Matrix)
    assert isinstance(S, Matrix)
    assert isinstance(Vh, Matrix)


def test_svd_reconstruction():
    U, S, Vh = svd_decomposition(A2)
    assert np.allclose(np.array(U.data) @ np.array(S.data) @ np.array(Vh.data), np.array(A2.data), atol=1e-8)


def test_eigen_decomposition_returns_vector_and_matrix():
    vals, vecs = eigen_decomposition(A2)
    assert isinstance(vals, Vector)
    assert isinstance(vecs, Matrix)


def test_eigen_decomposition_eigenvalues_count():
    vals, _ = eigen_decomposition(A2)
    assert len(vals.data) == 2


def test_cholesky_lower_triangular():
    L = cholesky_decomposition(SPD)
    assert isinstance(L, Matrix)
    l = np.array(L.data)
    assert np.allclose(l @ l.T, np.array(SPD.data), atol=1e-8)


def test_schur_decomposition():
    T, Z = schur_decomposition(A2)
    assert isinstance(T, Matrix)
    assert isinstance(Z, Matrix)


def test_polar_decomposition():
    P, U = polar_decomposition(A2)
    assert isinstance(P, Matrix)
    assert isinstance(U, Matrix)
