"""Tests for pyvelora/linalg/products.py."""
import numpy as np
import pytest
from pyvelora.core import Vector, Matrix
from pyvelora.linalg.products import dot, outer, matmul, matvec, cross, prod


def test_dot_product():
    u = Vector(np.array([1.0, 2.0, 3.0]))
    v = Vector(np.array([4.0, 5.0, 6.0]))
    assert np.isclose(dot(u, v), 32.0)


def test_outer_product_shape():
    u = Vector(np.array([1.0, 2.0]))
    v = Vector(np.array([3.0, 4.0, 5.0]))
    result = outer(u, v)
    assert isinstance(result, Matrix)
    assert result.shape == (2, 3)


def test_outer_product_values():
    u = Vector(np.array([1.0, 2.0]))
    v = Vector(np.array([3.0, 4.0]))
    result = outer(u, v)
    assert np.allclose(result.data, [[3, 4], [6, 8]])


def test_matmul_result():
    A = Matrix(np.array([[1.0, 2.0], [3.0, 4.0]]))
    B = Matrix(np.array([[5.0, 6.0], [7.0, 8.0]]))
    result = matmul(A, B)
    assert isinstance(result, Matrix)
    assert np.allclose(result.data, [[19, 22], [43, 50]])


def test_matmul_dimension_mismatch_raises():
    A = Matrix(np.ones((2, 3)))
    B = Matrix(np.ones((2, 2)))
    with pytest.raises(ValueError):
        matmul(A, B)


def test_matvec_result():
    A = Matrix(np.array([[1.0, 0.0], [0.0, 2.0]]))
    v = Vector(np.array([3.0, 4.0]))
    result = matvec(A, v)
    assert isinstance(result, Vector)
    assert np.allclose(result.data, [3.0, 8.0])


def test_matvec_dimension_mismatch_raises():
    A = Matrix(np.ones((2, 3)))
    v = Vector(np.ones(2))
    with pytest.raises(ValueError):
        matvec(A, v)


def test_cross_product():
    u = Vector(np.array([1.0, 0.0, 0.0]))
    v = Vector(np.array([0.0, 1.0, 0.0]))
    result = cross(u, v)
    assert isinstance(result, Vector)
    assert np.allclose(result.data, [0.0, 0.0, 1.0])


def test_cross_non_3d_raises():
    u = Vector(np.array([1.0, 2.0]))
    v = Vector(np.array([3.0, 4.0]))
    with pytest.raises(ValueError):
        cross(u, v)


def test_prod_along_axis0():
    A = Matrix(np.array([[1.0, 2.0], [3.0, 4.0]]))
    result = prod(A, axis=0)
    assert isinstance(result, Vector)
    assert np.allclose(result.data, [3.0, 8.0])


def test_prod_along_axis1():
    A = Matrix(np.array([[1.0, 2.0], [3.0, 4.0]]))
    result = prod(A, axis=1)
    assert np.allclose(result.data, [2.0, 12.0])


def test_prod_invalid_axis_raises():
    A = Matrix(np.ones((2, 2)))
    with pytest.raises(ValueError):
        prod(A, axis=2)
