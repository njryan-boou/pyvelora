import numpy as np

from pyvelora.core.array_base import Base, format_array, format_complex, format_float


class DemoArray(Base):
    pass


def test_format_float_trims_trailing_zeros():
    assert format_float(3.0) == "3"
    assert format_float(3.25) == "3.25"


def test_format_complex_prints_explicit_sign_and_j():
    assert format_complex(2 + 3j) == "2 + 3j"
    assert format_complex(2 - 3j) == "2 - 3j"


def test_format_array_uses_custom_float_formatting():
    arr = np.array([1.0, 2.5, 3.0])
    assert format_array(arr) == "[1 2.5 3]"


def test_array_base_stores_float64_numpy_data():
    arr = DemoArray([1, 2, 3])

    assert isinstance(arr.data, np.ndarray)
    assert arr.dtype == np.dtype("float64")
    assert np.array_equal(arr.data, np.array([1.0, 2.0, 3.0]))


def test_array_base_exposes_shape_ndim_size_dtype():
    arr = DemoArray([[1, 2], [3, 4]])

    assert arr.shape == (2, 2)
    assert arr.ndim == 2
    assert arr.size == 4
    assert arr.dtype == np.dtype("float64")


def test_array_base_mutation_via_data_is_supported():
    arr = DemoArray([1, 2, 3])

    arr.data[1] = 9

    assert np.array_equal(arr.data, np.array([1.0, 9.0, 3.0]))
