"""Tests for pyvelora/linalg/matrix_functions.py."""
import numpy as np
import pytest
from pyvelora.core import Matrix
from pyvelora.linalg.matrix_functions import matrix_exponential, matrix_power


def test_matrix_exponential_returns_matrix():
    A = Matrix(np.zeros((2, 2)))
    result = matrix_exponential(A)
    assert isinstance(result, Matrix)


def test_matrix_exponential_of_zero_is_identity():
    A = Matrix(np.zeros((3, 3)))
    result = matrix_exponential(A)
    assert np.allclose(result.data, np.eye(3), atol=1e-10)


def test_matrix_exponential_not_matrix_raises():
    with pytest.raises(TypeError):
        matrix_exponential(np.zeros((2, 2)))


def test_matrix_power_zero():
    A = Matrix(np.array([[2.0, 1.0], [0.0, 3.0]]))
    result = matrix_power(A, 0)
    assert isinstance(result, Matrix)
    assert np.allclose(result.data, np.eye(2), atol=1e-10)


def test_matrix_power_one():
    A = Matrix(np.array([[2.0, 1.0], [0.0, 3.0]]))
    result = matrix_power(A, 1)
    assert np.allclose(result.data, A.data, atol=1e-10)


def test_matrix_power_two():
    A = Matrix(np.array([[1.0, 1.0], [0.0, 1.0]]))
    result = matrix_power(A, 2)
    expected = np.array([[1.0, 2.0], [0.0, 1.0]])
    assert np.allclose(result.data, expected, atol=1e-10)


def test_matrix_power_non_square_raises():
    B = Matrix(np.ones((2, 3)))
    with pytest.raises(ValueError):
        matrix_power(B, 2)


def test_matrix_power_negative_exponent():
    A = Matrix(np.array([[2.0, 0.0], [0.0, 4.0]]))
    result = matrix_power(A, -1)
    expected = np.array([[0.5, 0.0], [0.0, 0.25]])
    assert np.allclose(result.data, expected, atol=1e-10)


def test_matrix_power_negative_exponent_singular_raises():
    A = Matrix(np.array([[1.0, 0.0], [0.0, 0.0]]))
    with pytest.raises(ValueError):
        matrix_power(A, -1)
