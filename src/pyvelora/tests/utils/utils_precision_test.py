"""Tests for pyvelora/utils/precision.py."""
import numpy as np
import pytest
from pyvelora.utils.precision import (
    isclose, allclose, is_zero, is_close, is_integer, round_small, round_to,
    clean, set_precision, get_precision, zero_threshold,
)


# ---------------------------------------------------------------------------
# isclose
# ---------------------------------------------------------------------------

def test_isclose_returns_true_for_identical():
    assert bool(isclose(1.0, 1.0)) is True


def test_isclose_returns_false_for_different():
    assert bool(isclose(1.0, 2.0)) is False


def test_isclose_elementwise_numpy():
    a = np.array([1.0, 2.0])
    b = np.array([1.0, 2.000001])
    result = isclose(a, b)
    assert result.shape == (2,)


def test_isclose_flat_list():
    result = isclose([1.0, 2.0], [1.0, 2.000001])
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0] is True


def test_isclose_nested_list():
    result = isclose([[1.0, 2.0], [3.0, 4.0]], [[1.0, 2.0], [3.0, 4.0]])
    assert result == [[True, True], [True, True]]


def test_isclose_nested_list_false():
    result = isclose([[1.0, 99.0]], [[1.0, 2.0]])
    assert result[0][1] is False


# ---------------------------------------------------------------------------
# allclose
# ---------------------------------------------------------------------------

def test_allclose_returns_bool():
    assert isinstance(allclose(1.0, 1.0), bool)


def test_allclose_true_numpy():
    assert allclose(np.array([1.0, 2.0]), np.array([1.0, 2.0])) is True


def test_allclose_false_numpy():
    assert allclose(np.array([1.0, 2.0]), np.array([1.0, 3.0])) is False


def test_allclose_true_nested_list():
    assert allclose([[1.0, 2.0], [3.0, 4.0]], [[1.0, 2.0], [3.0, 4.0]]) is True


def test_allclose_false_nested_list():
    assert allclose([[1.0, 2.0], [3.0, 99.0]], [[1.0, 2.0], [3.0, 4.0]]) is False


# ---------------------------------------------------------------------------
# is_zero
# ---------------------------------------------------------------------------

def test_is_zero_for_zero():
    assert bool(is_zero(0.0)) is True


def test_is_zero_for_small_value():
    assert bool(is_zero(1e-11)) is True


def test_is_zero_for_non_zero():
    assert bool(is_zero(1.0)) is False


def test_is_zero_nested_list():
    result = is_zero([[0.0, 1e-11], [1.0, 0.0]])
    assert result == [[True, True], [False, True]]


# ---------------------------------------------------------------------------
# is_close alias
# ---------------------------------------------------------------------------

def test_is_close_alias():
    assert bool(is_close(1.0, 1.0)) is True


# ---------------------------------------------------------------------------
# round_small / clean
# ---------------------------------------------------------------------------

def test_round_small_scalar_tiny():
    assert round_small(1e-15) == 0.0


def test_round_small_scalar_large():
    assert round_small(1.0) == 1.0


def test_round_small_flat_list():
    result = round_small([1e-15, 1.0])
    assert result == [0.0, 1.0]


def test_round_small_nested_list():
    result = round_small([[1e-15, 2.0], [0.0, 3.0]])
    assert result == [[0.0, 2.0], [0.0, 3.0]]


def test_clean_alias_scalar():
    assert clean(1e-15) == 0.0


def test_clean_alias_nested():
    result = clean([[1e-15, 2.0]])
    assert result == [[0.0, 2.0]]


# ---------------------------------------------------------------------------
# round_to
# ---------------------------------------------------------------------------

def test_round_to_scalar():
    assert round_to(3.14159, 2) == 3.14


def test_round_to_flat_list():
    result = round_to([1.005, 2.555], 2)
    assert result[1] == pytest.approx(2.56, abs=0.01)


def test_round_to_nested_list():
    result = round_to([[1.1111, 2.2222], [3.3333, 4.4444]], 2)
    assert result == [[1.11, 2.22], [3.33, 4.44]]


def test_round_to_complex():
    result = round_to(1.1111 + 2.2222j, 2)
    assert result == pytest.approx(1.11 + 2.22j, abs=1e-9)


def test_round_to_invalid_digits():
    with pytest.raises(ValueError):
        round_to(1.0, -1)


# ---------------------------------------------------------------------------
# is_integer
# ---------------------------------------------------------------------------

def test_is_integer_true():
    assert is_integer(2.0) is True


def test_is_integer_false():
    assert is_integer(2.5) is False


def test_is_integer_near_int():
    assert is_integer(3.0 + 1e-12) is True


def test_is_integer_flat_list():
    result = is_integer([1.0, 1.5, 2.0])
    assert result == [True, False, True]


def test_is_integer_nested_list():
    result = is_integer([[1.0, 1.5], [2.0, 2.0]])
    assert result == [[True, False], [True, True]]


# ---------------------------------------------------------------------------
# set_precision / get_precision
# ---------------------------------------------------------------------------

def test_set_precision_updates_threshold():
    original = get_precision()["zero_threshold"]
    set_precision(zero_tol=1e-6)
    assert get_precision()["zero_threshold"] == 1e-6
    set_precision(zero_tol=original)


def test_set_precision_negative_raises():
    with pytest.raises(ValueError):
        set_precision(zero_tol=-1.0)


def test_get_precision_returns_dict():
    assert "zero_threshold" in get_precision()


def test_zero_threshold_default():
    assert zero_threshold == 1e-10
