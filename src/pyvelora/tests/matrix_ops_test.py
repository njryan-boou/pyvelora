import numpy as np
import pytest

from pyvelora import Matrix, Vector
from pyvelora.linalg import (
    determinant,
    eigenvalues,
    eigenvectors,
    inverse,
    matrix_power,
    solve,
    trace,
    transpose,
)


def test_matrix_ops_transpose_returns_matrix():
    m = Matrix([[1, 2], [3, 4]])
    result = transpose(m)

    assert isinstance(result, Matrix)
    assert np.array_equal(result.data, np.array([[1, 3], [2, 4]]))


def test_matrix_ops_determinant_inverse_and_trace():
    m = Matrix([[4, 7], [2, 6]])

    assert np.isclose(determinant(m), 10.0)
    assert np.isclose(trace(m), 10.0)
    assert np.allclose(inverse(m).data, np.linalg.inv(m.data))


def test_matrix_ops_eigenvalues_and_eigenvectors():
    m = Matrix([[2, 0], [0, 3]])

    values = eigenvalues(m)
    eigvals, eigvecs = eigenvectors(m)

    assert np.allclose(np.sort(values), np.array([2.0, 3.0]))
    assert isinstance(eigvals, Vector)
    assert isinstance(eigvecs, Matrix)
    assert eigvals.shape == (2,)
    assert eigvecs.shape == (2, 2)


def test_matrix_ops_solve_linear_system():
    a = Matrix([[2, 1], [5, 7]])
    b = Vector([11, 13])

    x = solve(a, b)

    assert isinstance(x, Vector)
    assert np.allclose(x.data, np.linalg.solve(a.data, b.data))


def test_matrix_ops_matrix_power_integer_and_float():
    m = Matrix([[2, 0], [0, 3]])

    squared = matrix_power(m, 2)
    rooted = matrix_power(m, 0.5)

    assert isinstance(squared, Matrix)
    assert np.array_equal(squared.data, np.array([[4.0, 0.0], [0.0, 9.0]]))
    assert isinstance(rooted, Matrix)
    assert np.allclose(rooted.data, np.array([[np.sqrt(2.0), 0.0], [0.0, np.sqrt(3.0)]]))


def test_matrix_ops_type_checks():
    with pytest.raises(TypeError, match="Matrix objects"):
        transpose([[1, 2], [3, 4]])

    with pytest.raises(TypeError, match="Matrix objects"):
        determinant([[1, 2], [3, 4]])

    with pytest.raises(TypeError, match="Matrix objects"):
        matrix_power([[1, 2], [3, 4]], 2)
