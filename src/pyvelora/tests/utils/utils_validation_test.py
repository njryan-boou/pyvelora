"""Tests for pyvelora/utils/validation.py."""
import math
import numpy as np
import pytest
from pyvelora.core import Vector, Matrix, Tensor
from pyvelora.utils.validation import (
    require_vector, require_matrix, require_tensor,
    require_same_shape, require_square, require_dimension,
    isscalar, require_scalar, require_nonzero, require_positive,
    require_nonnegative, require_integer, require_real,
    require_complex, require_finite,
)


def test_require_vector_passes():
    require_vector(Vector(np.array([1.0, 2.0])))  # no exception


def test_require_vector_raises():
    with pytest.raises(TypeError):
        require_vector(Matrix(np.eye(2)))


def test_require_vector_custom_message():
    with pytest.raises(TypeError, match="need vector"):
        require_vector(1.0, message="need vector")


def test_require_matrix_passes():
    require_matrix(Matrix(np.eye(2)))  # no exception


def test_require_matrix_raises():
    with pytest.raises(TypeError):
        require_matrix(Vector(np.array([1.0])))


def test_require_tensor_passes():
    require_tensor(Tensor(np.ones((2, 2, 2))))  # no exception


def test_require_tensor_raises():
    with pytest.raises(TypeError):
        require_tensor(Matrix(np.eye(2)))


def test_require_same_shape_passes():
    a = Matrix(np.eye(2))
    b = Matrix(np.zeros((2, 2)))
    require_same_shape(a, b)  # no exception


def test_require_same_shape_raises():
    a = Matrix(np.eye(2))
    b = Matrix(np.ones((2, 3)))
    with pytest.raises(ValueError):
        require_same_shape(a, b)


def test_require_square_passes():
    require_square(Matrix(np.eye(3)))  # no exception


def test_require_square_raises():
    with pytest.raises(ValueError):
        require_square(Matrix(np.ones((2, 3))))


def test_require_dimension_passes():
    v = Vector(np.array([1.0, 2.0]))
    require_dimension(v, 1)  # no exception


def test_require_dimension_raises():
    v = Vector(np.array([1.0, 2.0]))
    with pytest.raises(ValueError):
        require_dimension(v, 2)


# ---------------------------------------------------------------------------
# isscalar
# ---------------------------------------------------------------------------

def test_isscalar_int():
    assert isscalar(1) is True


def test_isscalar_float():
    assert isscalar(1.0) is True


def test_isscalar_complex():
    assert isscalar(1+2j) is True


def test_isscalar_list_false():
    assert isscalar([1, 2]) is False


def test_isscalar_vector_false():
    assert isscalar(Vector(np.array([1.0]))) is False


# ---------------------------------------------------------------------------
# require_scalar
# ---------------------------------------------------------------------------

def test_require_scalar_passes():
    require_scalar(3.14)


def test_require_scalar_raises_for_list():
    with pytest.raises(TypeError):
        require_scalar([1, 2])


def test_require_scalar_raises_for_vector():
    with pytest.raises(TypeError):
        require_scalar(Vector(np.array([1.0])))


# ---------------------------------------------------------------------------
# require_nonzero
# ---------------------------------------------------------------------------

def test_require_nonzero_scalar_passes():
    require_nonzero(1.0)


def test_require_nonzero_scalar_raises():
    with pytest.raises(ValueError):
        require_nonzero(0)


def test_require_nonzero_vector_passes():
    require_nonzero(Vector(np.array([1.0, 2.0])))


def test_require_nonzero_vector_raises():
    with pytest.raises(ValueError):
        require_nonzero(Vector(np.array([1.0, 0.0])))


def test_require_nonzero_matrix_raises():
    with pytest.raises(ValueError):
        require_nonzero(Matrix(np.array([[1.0, 0.0], [2.0, 3.0]])))


# ---------------------------------------------------------------------------
# require_positive
# ---------------------------------------------------------------------------

def test_require_positive_scalar_passes():
    require_positive(1.0)


def test_require_positive_scalar_raises_zero():
    with pytest.raises(ValueError):
        require_positive(0.0)


def test_require_positive_scalar_raises_negative():
    with pytest.raises(ValueError):
        require_positive(-1.0)


def test_require_positive_matrix_raises():
    with pytest.raises(ValueError):
        require_positive(Matrix(np.array([[1.0, -1.0], [2.0, 3.0]])))


# ---------------------------------------------------------------------------
# require_nonnegative
# ---------------------------------------------------------------------------

def test_require_nonnegative_zero_passes():
    require_nonnegative(0.0)


def test_require_nonnegative_positive_passes():
    require_nonnegative(1.0)


def test_require_nonnegative_scalar_raises():
    with pytest.raises(ValueError):
        require_nonnegative(-1.0)


def test_require_nonnegative_matrix_raises():
    with pytest.raises(ValueError):
        require_nonnegative(Matrix(np.array([[1.0, -0.5], [2.0, 3.0]])))


# ---------------------------------------------------------------------------
# require_integer
# ---------------------------------------------------------------------------

def test_require_integer_passes():
    require_integer(2.0)


def test_require_integer_raises():
    with pytest.raises(ValueError):
        require_integer(2.5)


def test_require_integer_vector_passes():
    require_integer(Vector(np.array([1.0, 2.0, 3.0])))


def test_require_integer_vector_raises():
    with pytest.raises(ValueError):
        require_integer(Vector(np.array([1.0, 1.5])))


# ---------------------------------------------------------------------------
# require_real
# ---------------------------------------------------------------------------

def test_require_real_float_passes():
    require_real(1.0)


def test_require_real_complex_raises():
    with pytest.raises(ValueError):
        require_real(1+2j)


def test_require_real_pure_real_complex_passes():
    require_real(complex(3.0, 0.0))  # imag == 0 is considered real


# ---------------------------------------------------------------------------
# require_complex
# ---------------------------------------------------------------------------

def test_require_complex_passes():
    require_complex(1+2j)


def test_require_complex_scalar_raises():
    with pytest.raises(ValueError):
        require_complex(1.0)


# ---------------------------------------------------------------------------
# require_finite
# ---------------------------------------------------------------------------

def test_require_finite_passes():
    require_finite(1.0)


def test_require_finite_raises_inf():
    with pytest.raises(ValueError):
        require_finite(math.inf)


def test_require_finite_raises_nan():
    with pytest.raises(ValueError):
        require_finite(math.nan)


def test_require_finite_matrix_raises():
    with pytest.raises(ValueError):
        require_finite(Matrix(np.array([[1.0, math.inf], [2.0, 3.0]])))
