"""Tests for pyvelora/linalg/constructors.py."""
import numpy as np
import pytest
from pyvelora.core import Vector, Matrix
from pyvelora.linalg.constructors import (
    zeros, ones, full, identity, diagonal, from_rows, from_cols,
)


def test_zeros_shape():
    A = zeros(2, 3)
    assert isinstance(A, Matrix)
    assert A.shape == (2, 3)
    assert np.allclose(A.data, np.zeros((2, 3)))


def test_ones_shape():
    A = ones(3, 2)
    assert isinstance(A, Matrix)
    assert np.allclose(A.data, np.ones((3, 2)))


def test_full_value():
    A = full(2, 2, 7.0)
    assert isinstance(A, Matrix)
    assert np.allclose(A.data, np.full((2, 2), 7.0))


def test_identity_is_eye():
    I = identity(3)
    assert isinstance(I, Matrix)
    assert np.allclose(I.data, np.eye(3))


def test_diagonal_from_list():
    D = diagonal([1.0, 2.0, 3.0])
    assert isinstance(D, Matrix)
    assert np.allclose(D.data, np.diag([1.0, 2.0, 3.0]))


def test_diagonal_from_vector():
    v = Vector(np.array([4.0, 5.0]))
    D = diagonal(v)
    assert np.allclose(D.data, np.diag([4.0, 5.0]))


def test_from_rows():
    rows = [Vector(np.array([1.0, 2.0])), Vector(np.array([3.0, 4.0]))]
    A = from_rows(rows)
    assert isinstance(A, Matrix)
    assert np.allclose(A.data, [[1, 2], [3, 4]])


def test_from_cols():
    cols = [Vector(np.array([1.0, 3.0])), Vector(np.array([2.0, 4.0]))]
    A = from_cols(cols)
    assert isinstance(A, Matrix)
    assert np.allclose(A.data, [[1, 2], [3, 4]])


def test_zeros_invalid_type_raises():
    with pytest.raises(TypeError):
        zeros(2.0, 3)


def test_zeros_non_positive_raises():
    with pytest.raises(ValueError):
        zeros(0, 3)


def test_diagonal_invalid_type_raises():
    with pytest.raises(TypeError):
        diagonal(np.array([1, 2]))
