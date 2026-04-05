"""Tests for pyvelora/diffyq/utils.py."""
import numpy as np
import pytest
from pyvelora.diffyq.utils import _is_scalar, _to_list, _from_list


def test_is_scalar_int():
    assert _is_scalar(3) is True


def test_is_scalar_float():
    assert _is_scalar(3.14) is True


def test_is_scalar_list():
    assert _is_scalar([1, 2]) is False


def test_is_scalar_array():
    assert _is_scalar(np.array([1.0])) is False


def test_to_list_scalar():
    assert _to_list(5.0) == [5.0]


def test_to_list_list():
    assert _to_list([1, 2, 3]) == [1, 2, 3]


def test_to_list_array():
    result = _to_list(np.array([4.0, 5.0]))
    assert list(result) == [4.0, 5.0]


def test_from_list_scalar_y0():
    arr = np.array([7.0])
    assert _from_list(1.0, arr) == 7.0


def test_from_list_vector_y0():
    y0 = [1, 2]
    arr = np.array([3.0, 4.0])
    result = _from_list(y0, arr)
    assert np.allclose(result, arr)
