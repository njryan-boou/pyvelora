"""Tests for pyvelora/utils/errors.py."""
import pytest
from pyvelora.utils.errors import PyveloraError, ShapeError, DimensionError


def test_pyvelora_error_is_exception():
    assert issubclass(PyveloraError, Exception)


def test_shape_error_is_pyvelora_error():
    assert issubclass(ShapeError, PyveloraError)


def test_dimension_error_is_pyvelora_error():
    assert issubclass(DimensionError, PyveloraError)


def test_raise_pyvelora_error():
    with pytest.raises(PyveloraError):
        raise PyveloraError("base error")


def test_raise_shape_error():
    with pytest.raises(ShapeError):
        raise ShapeError("bad shape")


def test_raise_dimension_error():
    with pytest.raises(DimensionError):
        raise DimensionError("bad dimension")


def test_shape_error_caught_as_pyvelora_error():
    with pytest.raises(PyveloraError):
        raise ShapeError("shape mismatch")


def test_dimension_error_caught_as_pyvelora_error():
    with pytest.raises(PyveloraError):
        raise DimensionError("dimension mismatch")


def test_error_message_preserved():
    msg = "specific message"
    err = ShapeError(msg)
    assert str(err) == msg


# ---------------------------------------------------------------------------
# SingularMatrixError
# ---------------------------------------------------------------------------

from pyvelora.utils.errors import SingularMatrixError, ConvergenceError, DomainError


def test_singular_matrix_error_is_shape_error():
    assert issubclass(SingularMatrixError, ShapeError)


def test_singular_matrix_error_is_pyvelora_error():
    assert issubclass(SingularMatrixError, PyveloraError)


def test_raise_singular_matrix_error():
    with pytest.raises(SingularMatrixError):
        raise SingularMatrixError("matrix is singular")


def test_singular_matrix_error_caught_as_shape_error():
    with pytest.raises(ShapeError):
        raise SingularMatrixError("singular")


# ---------------------------------------------------------------------------
# ConvergenceError
# ---------------------------------------------------------------------------

def test_convergence_error_is_pyvelora_error():
    assert issubclass(ConvergenceError, PyveloraError)


def test_raise_convergence_error():
    with pytest.raises(ConvergenceError):
        raise ConvergenceError("did not converge")


def test_convergence_error_caught_as_pyvelora_error():
    with pytest.raises(PyveloraError):
        raise ConvergenceError("failed")


# ---------------------------------------------------------------------------
# DomainError
# ---------------------------------------------------------------------------

def test_domain_error_is_pyvelora_error():
    assert issubclass(DomainError, PyveloraError)


def test_raise_domain_error():
    with pytest.raises(DomainError):
        raise DomainError("log of negative")


def test_domain_error_caught_as_pyvelora_error():
    with pytest.raises(PyveloraError):
        raise DomainError("bad domain")
