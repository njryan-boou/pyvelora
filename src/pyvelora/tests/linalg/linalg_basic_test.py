"""Tests for pyvelora/linalg/basic.py."""
import numpy as np
import pytest
from pyvelora.core import Matrix, Vector, Tensor
from pyvelora.linalg.basic import (
    get_row, get_col, swap_rows, swap_cols,
    transpose, add, subtract, scalar_multiply, hamard_product,
)


M = Matrix(np.array([[1.0, 2.0], [3.0, 4.0]]))


def test_get_row():
    r = get_row(M, 0)
    assert isinstance(r, Vector)
    assert np.allclose(r.data, [1.0, 2.0])


def test_get_row_invalid_index():
    with pytest.raises(ValueError):
        get_row(M, 5)


def test_get_col():
    c = get_col(M, 1)
    assert isinstance(c, Vector)
    assert np.allclose(c.data, [2.0, 4.0])


def test_swap_rows():
    result = swap_rows(M, 0, 1)
    assert np.allclose(result.data[0], [3.0, 4.0])
    assert np.allclose(result.data[1], [1.0, 2.0])


def test_swap_cols():
    result = swap_cols(M, 0, 1)
    result_arr = np.array(result.data)
    assert np.allclose(result_arr[:, 0], [2.0, 4.0])
    assert np.allclose(result_arr[:, 1], [1.0, 3.0])


def test_transpose():
    result = transpose(M)
    assert np.allclose(result.data, np.array(M.data).T)


def test_add_matrix():
    result = add(M, M)
    assert np.allclose(result.data, np.array(M.data) * 2)


def test_subtract_matrix():
    result = subtract(M, M)
    assert np.allclose(result.data, np.zeros((2, 2)))


def test_add_vector():
    v = Vector([1.0, 2.0])
    result = add(v, v)
    assert isinstance(result, Vector)
    assert np.allclose(result.data, [2.0, 4.0])


def test_scalar_multiply_matrix():
    result = scalar_multiply(M, 3.0)
    assert np.allclose(result.data, np.array(M.data) * 3)


def test_scalar_multiply_vector():
    v = Vector([1.0, 2.0, 3.0])
    result = scalar_multiply(v, 2.0)
    assert np.allclose(result.data, [2.0, 4.0, 6.0])


def test_hamard_product():
    result = hamard_product(M, M)
    assert np.allclose(result.data, np.array(M.data) ** 2)
