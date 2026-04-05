"""Tests for pyvelora/utils/numpy_utils.py."""
import numpy as np
import pytest
from pyvelora.utils.numpy_utils import (
    linspace, logspace, arange,
    meshgrid, linspace_nd, logspace_nd, arange_nd,
)


# ---------------------------------------------------------------------------
# linspace
# ---------------------------------------------------------------------------

def test_linspace_basic():
    result = linspace(0.0, 1.0, 5)
    assert len(result) == 5
    assert result[0] == pytest.approx(0.0)
    assert result[-1] == pytest.approx(1.0)


def test_linspace_single():
    result = linspace(3.0, 7.0, 1)
    assert result == [3.0]


def test_linspace_evenly_spaced():
    result = linspace(0.0, 4.0, 5)
    assert result == pytest.approx([0.0, 1.0, 2.0, 3.0, 4.0])


def test_linspace_invalid_num_raises():
    with pytest.raises(ValueError):
        linspace(0.0, 1.0, 0)


# ---------------------------------------------------------------------------
# logspace
# ---------------------------------------------------------------------------

def test_logspace_basic():
    result = logspace(0.0, 2.0, 3)
    assert len(result) == 3
    assert result[0] == pytest.approx(1.0)    # 10^0
    assert result[1] == pytest.approx(10.0)   # 10^1
    assert result[2] == pytest.approx(100.0)  # 10^2


def test_logspace_single():
    result = logspace(2.0, 4.0, 1)
    assert result == pytest.approx([100.0])


# ---------------------------------------------------------------------------
# arange
# ---------------------------------------------------------------------------

def test_arange_basic():
    result = arange(0.0, 3.0, 1.0)
    assert result == pytest.approx([0.0, 1.0, 2.0])


def test_arange_fractional_step():
    result = arange(0.0, 1.0, 0.5)
    assert result == pytest.approx([0.0, 0.5])


def test_arange_negative_step():
    result = arange(3.0, 0.0, -1.0)
    assert result == pytest.approx([3.0, 2.0, 1.0])


def test_arange_zero_step_raises():
    with pytest.raises(ValueError):
        arange(0.0, 1.0, 0.0)


def test_meshgrid_1d():
    x = np.array([1.0, 2.0])
    y = np.array([3.0, 4.0])
    g = meshgrid(x, y)
    assert len(g) == 2
    assert g[0].shape == (2, 2)


def test_meshgrid_uses_ij_indexing():
    x = np.array([1.0, 2.0])
    y = np.array([3.0, 4.0, 5.0])
    gx, gy = meshgrid(x, y)
    # ij indexing: first index varies along rows
    assert gx.shape == (2, 3)
    assert gy.shape == (2, 3)


def test_linspace_nd_1d_returns_single_array():
    result = linspace_nd(0.0, 1.0, num=5, ndim=1)
    assert len(result) == 1
    assert result[0].shape == (5,)


def test_linspace_nd_2d_returns_two_grids():
    result = linspace_nd(0.0, 1.0, num=4, ndim=2)
    assert len(result) == 2
    assert result[0].shape == (4, 4)


def test_linspace_nd_invalid_ndim_raises():
    with pytest.raises(ValueError):
        linspace_nd(0.0, 1.0, num=5, ndim=0)


def test_logspace_nd_1d():
    result = logspace_nd(0.0, 2.0, num=3, ndim=1)
    assert len(result) == 1
    assert np.isclose(result[0][0], 1.0)  # 10^0 = 1


def test_logspace_nd_invalid_ndim_raises():
    with pytest.raises(ValueError):
        logspace_nd(0.0, 1.0, ndim=-1)


def test_arange_nd_1d():
    result = arange_nd(0.0, 3.0, 1.0, ndim=1)
    assert len(result) == 1
    assert list(result[0]) == [0.0, 1.0, 2.0]


def test_arange_nd_2d():
    result = arange_nd(0.0, 2.0, 1.0, ndim=2)
    assert len(result) == 2
    assert result[0].shape == (2, 2)


def test_arange_nd_invalid_ndim_raises():
    with pytest.raises(ValueError):
        arange_nd(0.0, 3.0, 1.0, ndim=0)
