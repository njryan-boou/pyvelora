"""Tests for pyvelora/linalg/norms.py."""
import numpy as np
import pytest
from pyvelora.core import Vector, Matrix
from pyvelora.linalg.norms import (
    vector_norm, frobenius_norm, one_norm, inf_norm, normalize,
)


def test_vector_norm_l2():
    v = Vector(np.array([3.0, 4.0]))
    assert np.isclose(vector_norm(v), 5.0)


def test_vector_norm_l1():
    v = Vector(np.array([3.0, -4.0]))
    assert np.isclose(vector_norm(v, p=1), 7.0)


def test_vector_norm_invalid_p_raises():
    v = Vector(np.array([1.0, 2.0]))
    with pytest.raises(ValueError):
        vector_norm(v, p=0)


def test_vector_norm_wrong_type_raises():
    with pytest.raises(TypeError):
        vector_norm(np.array([1.0, 2.0]))


def test_frobenius_norm():
    A = Matrix(np.array([[1.0, 2.0], [3.0, 4.0]]))
    expected = np.linalg.norm(A.data, ord="fro")
    assert np.isclose(frobenius_norm(A), expected)


def test_frobenius_norm_wrong_type_raises():
    with pytest.raises(TypeError):
        frobenius_norm(np.eye(2))


def test_one_norm():
    A = Matrix(np.array([[1.0, 2.0], [3.0, 4.0]]))
    expected = float(np.linalg.norm(A.data, ord=1))
    assert np.isclose(one_norm(A), expected)


def test_inf_norm():
    A = Matrix(np.array([[1.0, 2.0], [3.0, 4.0]]))
    expected = float(np.linalg.norm(A.data, ord=np.inf))
    assert np.isclose(inf_norm(A), expected)


def test_normalize_unit_length():
    v = Vector(np.array([3.0, 4.0]))
    n = normalize(v)
    assert isinstance(n, Vector)
    assert np.isclose(np.linalg.norm(n.data), 1.0)


def test_normalize_zero_vector_raises():
    v = Vector(np.zeros(3))
    with pytest.raises(ValueError):
        normalize(v)
