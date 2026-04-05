"""Module for pyvelora/tests/vector_test.py."""

import numpy as np
import pytest

from pyvelora import Vector


def test_vector_init_valid_1d_data():
	v = Vector([1, 2, 3])
	assert isinstance(v, Vector)
	assert v.shape == (3,)
	assert v.ndim == 1
	assert v.size == 3


def test_vector_init_raises_for_non_1d_data():
	with pytest.raises(ValueError, match="Vector must be 1D"):
		Vector([[1, 2], [3, 4]])


def test_vector_getitem_scalar_returns_scalar():
	v = Vector([10, 20, 30])
	assert v[1] == 20


def test_vector_getitem_slice_returns_numpy_array():
	v = Vector([10, 20, 30, 40])
	sliced = v[1:3]
	assert isinstance(sliced, list)
	assert np.array_equal(sliced, np.array([20.0, 30.0]))


def test_vector_setitem_updates_value():
	v = Vector([1, 2, 3])
	v[1] = 99
	assert v[1] == 99


def test_vector_len_iter_and_contains():
	v = Vector([4, 5, 6])
	assert len(v) == 3
	assert list(v) == [4, 5, 6]
	assert 5 in v
	assert 9 not in v


def test_vector_bool_true_for_non_empty_false_for_empty():
	assert bool(Vector([1])) is True
	assert bool(Vector([])) is False


def test_vector_add_and_subtract():
	v1 = Vector([1, 2, 3])
	v2 = Vector([4, 5, 6])

	summed = v1 + v2
	diff = v2 - v1

	assert isinstance(summed, Vector)
	assert isinstance(diff, Vector)
	assert np.array_equal(summed.data, np.array([5, 7, 9]))
	assert np.array_equal(diff.data, np.array([3, 3, 3]))


def test_vector_mul_and_div_elementwise():
	v1 = Vector([2, 4, 6])
	v2 = Vector([1, 2, 3])
	multiplied = v1 * v2
	divided = v1 / v2

	assert np.allclose(multiplied.data, np.array([2.0, 8.0, 18.0]))
	assert np.allclose(divided.data, np.array([2.0, 2.0, 2.0]))


def test_vector_repr_and_str():
	v = Vector([1, 2, 3])
	assert repr(v).startswith("Vector(")
	assert str(v) == "[1 2 3]"
	assert np.allclose(v.data, np.array([1, 2, 3]))


def test_vector_radd_rsub_rmul():
	v1 = Vector([1, 2, 3])
	v2 = Vector([4, 5, 6])

	radd_result = v1.__radd__(v2)
	rsub_result = v1.__rsub__(v2)
	rmul_result = v1.__rmul__(v2)

	assert isinstance(radd_result, Vector)
	assert isinstance(rsub_result, Vector)
	assert isinstance(rmul_result, Vector)
	assert np.allclose(radd_result.data, np.array([5, 7, 9]))
	assert np.allclose(rsub_result.data, np.array([3, 3, 3]))
	assert np.allclose(rmul_result.data, np.array([4, 10, 18]))


def test_vector_rtruediv():
	v1 = Vector([2, 4, 6])
	v2 = Vector([4, 8, 12])

	rtruediv_result = v1.__rtruediv__(v2)

	assert isinstance(rtruediv_result, Vector)
	assert np.allclose(rtruediv_result.data, np.array([2.0, 2.0, 2.0]))


def test_vector_iadd_isub_imul():
	v1 = Vector([1, 2, 3])
	v2 = Vector([4, 5, 6])
	orig_id = id(v1.data)

	result = v1.__iadd__(v2)
	assert result is v1
	assert np.allclose(v1.data, np.array([5, 7, 9]))

	v1 = Vector([10, 15, 20])
	v2 = Vector([2, 3, 4])
	result = v1.__isub__(v2)
	assert result is v1
	assert np.allclose(v1.data, np.array([8, 12, 16]))

	v1 = Vector([2, 3, 4])
	v2 = Vector([5, 6, 7])
	result = v1.__imul__(v2)
	assert result is v1
	assert np.allclose(v1.data, np.array([10, 18, 28]))


def test_vector_itruediv():
	v1 = Vector([10, 20, 30])
	v2 = Vector([2, 4, 5])

	result = v1.__itruediv__(v2)
	assert result is v1
	assert np.allclose(v1.data, np.array([5.0, 5.0, 6.0]))


def test_vector_polar_accepts_degrees_for_theta():
	v = Vector([2, 90], type="polar", degrees=True)
	assert np.allclose(v.data, np.array([0.0, 2.0]), atol=1e-9)


def test_vector_spherical_accepts_degrees_for_theta_phi():
	v = Vector([1, 90, 0], type="spherical", degrees=True)
	assert np.allclose(v.data, np.array([1.0, 0.0, 0.0]), atol=1e-9)


def test_vector_cylindrical_accepts_degrees_for_phi():
	v = Vector([2, 90, 3], type="cylindrical", degrees=True)
	assert np.allclose(v.data, np.array([0.0, 2.0, 3.0]), atol=1e-9)


# ---- coordinate system edge cases ----

def test_vector_polar_accepts_radians():
	v = Vector([1, np.pi / 2], type="polar", degrees=False)
	assert np.allclose(v.data, np.array([0.0, 1.0]), atol=1e-9)


def test_vector_polar_raises_for_wrong_size():
	with pytest.raises(ValueError, match="Polar coordinates"):
		Vector([1, 2, 3], type="polar")


def test_vector_spherical_raises_for_wrong_size():
	with pytest.raises(ValueError, match="Spherical coordinates"):
		Vector([1, 2], type="spherical")


def test_vector_cylindrical_raises_for_wrong_size():
	with pytest.raises(ValueError, match="Cylindrical coordinates"):
		Vector([1, 2], type="cylindrical")


def test_vector_cartesian_type_is_not_supported():
	with pytest.raises(ValueError, match="Unknown coordinate type"):
		Vector([3, 4], type="cartesian")


def test_vector_unknown_coordinate_type_raises():
	with pytest.raises(ValueError, match="Unknown coordinate type"):
		Vector([1, 2, 3], type="bogus")


# ---- magnitude and normalize ----

def test_vector_magnitude_pythagorean():
	v = Vector([3, 4])
	assert np.isclose(np.linalg.norm(v.data), 5.0)


def test_vector_magnitude_zero_vector():
	v = Vector([0, 0, 0])
	assert np.linalg.norm(v.data) == 0.0


def test_vector_normalize_returns_unit_vector():
	from pyvelora.linalg import normalize
	v = Vector([3, 0, 4])
	n = normalize(v)
	assert isinstance(n, Vector)
	assert np.isclose(np.linalg.norm(n.data), 1.0)
	assert np.allclose(n.data, np.array([0.6, 0.0, 0.8]))


def test_vector_normalize_raises_for_zero_vector():
	from pyvelora.linalg import normalize
	with pytest.raises(ValueError, match="Cannot normalize a zero vector"):
		normalize(Vector([0.0, 0.0]))


# ---- scalar conversion ----

def test_vector_float_for_single_element_vector():
	assert float(Vector([7])) == 7.0


def test_vector_float_raises_for_size_gt_1():
	with pytest.raises(ValueError, match="Cannot convert"):
		float(Vector([1, 2]))


def test_vector_complex_for_single_element_vector():
	assert complex(Vector([3])) == 3 + 0j


# ---- equality and ordering operators ----

def test_vector_eq_and_ne_elementwise():
	v1 = Vector([1, 2, 3])
	v2 = Vector([1, 2, 3])
	v3 = Vector([1, 9, 3])

	eq = v1 == v2
	ne = v1 != v3

	assert eq is True
	assert ne is True


def test_vector_lt_le_gt_ge():
	assert bool(Vector([1, 2]) < Vector([3, 4])) is True
	assert bool(Vector([1, 1]) <= Vector([1, 2])) is True
	assert bool(Vector([3, 4]) > Vector([1, 2])) is True
	assert bool(Vector([3, 3]) >= Vector([3, 2])) is True
	assert bool(Vector([1, 5]) < Vector([2, 4])) is False


def test_vector_comparison_with_wrong_type_returns_not_implemented():
	v = Vector([1, 2])
	assert Vector.__lt__(v, 42) is NotImplemented


# ---- shape / type guards ----

def test_vector_add_raises_for_shape_mismatch():
	with pytest.raises(ValueError, match="same size"):
		Vector([1, 2]) + Vector([1, 2, 3])


def test_vector_div_raises_for_zero_denominator():
	with pytest.raises(ValueError, match="Cannot divide by zero"):
		Vector([1, 2]) / Vector([0, 1])


def test_vector_arithmetic_with_wrong_type_raises_type_error():
	v = Vector([1, 2])
	with pytest.raises(TypeError):
		v.__add__(42)


# ---- array utility methods ----

def test_vector_astype_changes_underlying_numpy_dtype():
	v = Vector([1, 2, 3])
	# astype returns a copy of the VectorData with the same elements.
	raw = v.data.astype(np.int32)
	assert isinstance(raw, list)
	assert list(raw) == [1, 2, 3]


def test_vector_reshape_returns_correct_shape():
	v = Vector([1, 2, 3, 4, 5, 6])
	# VectorData.reshape returns a MatrixData for 2-D targets
	reshaped = v.data.reshape(2, 3)
	assert reshaped.shape == (2, 3)


def test_vector_flatten_returns_vector():
	v = Vector([1, 2, 3])
	flat = v.flatten()
	assert isinstance(flat, Vector)
	assert flat.shape == (3,)


def test_vector_copy_is_independent():
	v = Vector([1, 2, 3])
	c = v.copy()
	c[0] = 99
	assert v[0] == 1


def test_vector_deepcopy_is_independent():
	import copy
	v = Vector([1, 2, 3])
	c = copy.deepcopy(v)
	c[0] = 99
	assert v[0] == 1


# ---- property accessors ----

def test_vector_dtype_is_float64():
	v = Vector([1, 2, 3])
	assert v.dtype == np.dtype("float64")


def test_vector_ndim_is_1():
	v = Vector([1, 2, 3])
	assert v.ndim == 1


def test_vector_contains_float_close_to_stored_value():
	v = Vector([1.0, 1e-10, 3.0])
	assert 1.0 in v
	assert 99.0 not in v


def test_vector_helper_methods_delegate_through_linalg_ops():
	from pyvelora.linalg import dot, cross, vector_norm
	v1 = Vector([1, 2, 3])
	v2 = Vector([4, 5, 6])

	assert np.isclose(dot(v1, v2), 32.0)
	assert np.array_equal(cross(v1, v2).data, np.cross(v1.data, v2.data))
	assert np.isclose(vector_norm(v1), np.linalg.norm(v1.data))






