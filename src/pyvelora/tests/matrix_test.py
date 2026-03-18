import numpy as np
import pytest

from pyvelora import Matrix, Vector
from pyvelora.utils import ShapeError


def test_matrix_init_valid_2d_data():
	m = Matrix([[1, 2], [3, 4]])
	assert isinstance(m, Matrix)
	assert m.shape == (2, 2)
	assert m.ndim == 2
	assert m.size == 4


def test_matrix_init_raises_for_non_2d_data():
	with pytest.raises(ValueError, match="Matrix must be 2D"):
		Matrix([1, 2, 3])


def test_matrix_getitem_scalar_and_slice():
	m = Matrix([[1, 2], [3, 4]])
	assert m[0, 1] == 2

	row = m[0]
	assert isinstance(row, Vector)
	assert np.allclose(row.data, np.array([1, 2]))

	column = m[:, :1]
	assert isinstance(column, Matrix)
	assert np.allclose(column.data, np.array([[1], [3]]))


def test_matrix_setitem_updates_value():
	m = Matrix([[1, 2], [3, 4]])
	m[1, 0] = 9
	assert m[1, 0] == 9


def test_matrix_len_iter_and_contains():
	m = Matrix([[1, 2], [3, 4]])
	assert len(m) == 2

	rows = list(m)
	assert len(rows) == 2
	assert np.allclose(rows[0], np.array([1.0, 2.0]))
	assert 2 in m
	assert 99 not in m
	assert bool(m) is True


def test_matrix_add_and_subtract():
	m1 = Matrix([[1, 2], [3, 4]])
	m2 = Matrix([[5, 6], [7, 8]])

	added = m1 + m2
	subtracted = m2 - m1

	assert isinstance(added, Matrix)
	assert isinstance(subtracted, Matrix)
	assert np.allclose(added.data, np.array([[6, 8], [10, 12]]))
	assert np.allclose(subtracted.data, np.array([[4, 4], [4, 4]]))


def test_matrix_add_raises_for_shape_mismatch():
	m1 = Matrix([[1, 2], [3, 4]])
	m2 = Matrix([[1, 2, 3], [4, 5, 6]])

	with pytest.raises(ValueError, match="shapes must be the same"):
		_ = m1 + m2


def test_matrix_mul_and_div_elementwise():
	m1 = Matrix([[2, 4], [6, 8]])
	m2 = Matrix([[1, 2], [3, 4]])
	multiplied = m1 * m2
	divided = m1 / m2

	assert np.allclose(multiplied.data, np.array([[2, 8], [18, 32]]))
	assert np.allclose(divided.data, np.array([[2.0, 2.0], [2.0, 2.0]]))


def test_matrix_divide_by_zero_raises():
	m = Matrix([[1, 2], [3, 4]])
	zero_matrix = Matrix([[1, 0], [1, 1]])
	with pytest.raises(ZeroDivisionError, match="Cannot divide by zero"):
		_ = m / zero_matrix





def test_matrix_matmul_matrix_returns_matrix():
	m1 = Matrix([[1, 2], [3, 4]])
	m2 = Matrix([[5, 6], [7, 8]])
	result = m1 @ m2

	assert isinstance(result, Matrix)
	assert np.allclose(result.data, np.array([[19, 22], [43, 50]]))


def test_matrix_matmul_raises_for_misaligned_shapes():
	m1 = Matrix([[1, 2], [3, 4]])
	m2 = Matrix([[1, 2], [3, 4], [5, 6]])

	with pytest.raises(ValueError, match="not aligned for matrix multiplication"):
		_ = m1 @ m2


def test_matrix_comparison_operators_return_matrix():
	m1 = Matrix([[1, 5], [3, 4]])
	m2 = Matrix([[2, 5], [1, 6]])

	less_than = m1 < m2
	equal_to = m1 == m1.copy()

	assert isinstance(less_than, Matrix)
	assert isinstance(equal_to, Matrix)
	assert np.array_equal(less_than.data, np.array([[True, False], [False, True]]))
	assert np.array_equal(equal_to.data, np.array([[True, True], [True, True]]))


def test_matrix_unary_and_copy_methods():
	m = Matrix([[1, -2], [-3, 4]])

	assert np.allclose((-m).data, np.array([[-1, 2], [3, -4]]))
	assert np.allclose((+m).data, np.array([[1, -2], [-3, 4]]))
	assert np.allclose(abs(m).data, np.array([[1, 2], [3, 4]]))

	cloned = m.copy()
	assert isinstance(cloned, Matrix)
	assert cloned is not m
	assert np.allclose(cloned.data, m.data)


def test_matrix_str_omits_trailing_decimal_for_integer_values():
	m = Matrix([[1, 2], [3, 4]])
	assert str(m) == "[[1 2]\n [3 4]]"


def test_matrix_transpose_property():
	m = Matrix([[1, 2], [3, 4]])
	transposed = m.T

	assert isinstance(transposed, Matrix)
	assert np.allclose(transposed.data, np.array([[1, 3], [2, 4]]))


def test_matrix_determinant_trace_and_inverse():
	m = Matrix([[4, 7], [2, 6]])

	assert np.isclose(m.determinant(), 10.0)
	assert np.isclose(m.trace(), 10.0)
	assert np.allclose(m.inverse().data, np.linalg.inv(np.array([[4, 7], [2, 6]])))


def test_matrix_square_only_operations_raise_for_non_square():
	m = Matrix([[1, 2, 3], [4, 5, 6]])

	with pytest.raises(ValueError, match="square matrices"):
		m.determinant()

	with pytest.raises(ValueError, match="square matrices"):
		m.inverse()

	with pytest.raises(ValueError, match="square matrices"):
		m.trace()

	with pytest.raises(ValueError, match="square matrices"):
		m.eigenvalues()

	with pytest.raises(ValueError, match="square matrices"):
		m.eigenvectors()


def test_matrix_eigenvalues_and_eigenvectors_shapes():
	m = Matrix([[2, 0], [0, 3]])

	values = m.eigenvalues()
	eigenvalues, eigenvectors = m.eigenvectors()

	assert np.allclose(np.sort(values), np.array([2.0, 3.0]))
	assert isinstance(eigenvalues, Vector)
	assert isinstance(eigenvectors, Matrix)
	assert eigenvalues.shape == (2,)
	assert eigenvectors.shape == (2, 2)


def test_matrix_radd_rsub_rmul():
	m1 = Matrix([[1, 2], [3, 4]])
	m2 = Matrix([[5, 6], [7, 8]])

	radd_result = m1.__radd__(m2)
	rsub_result = m1.__rsub__(m2)
	rmul_result = m1.__rmul__(m2)

	assert isinstance(radd_result, Matrix)
	assert isinstance(rsub_result, Matrix)
	assert isinstance(rmul_result, Matrix)
	assert np.allclose(radd_result.data, np.array([[6, 8], [10, 12]]))
	assert np.allclose(rsub_result.data, np.array([[4, 4], [4, 4]]))
	assert np.allclose(rmul_result.data, np.array([[5, 12], [21, 32]]))


def test_matrix_rtruediv_rmatmul():
	m1 = Matrix([[2, 4], [6, 8]])
	m2 = Matrix([[4, 8], [12, 16]])

	rtruediv_result = m1.__rtruediv__(m2)
	assert isinstance(rtruediv_result, Matrix)
	assert np.allclose(rtruediv_result.data, np.array([[2.0, 2.0], [2.0, 2.0]]))

	m3 = Matrix([[1, 2], [3, 4]])
	m4 = Matrix([[5, 6], [7, 8]])
	rmatmul_result = m3.__rmatmul__(m4)
	assert isinstance(rmatmul_result, Matrix)
	# m4 @ m3 = [[5,6],[7,8]] @ [[1,2],[3,4]] = [[5*1+6*3, 5*2+6*4], [7*1+8*3, 7*2+8*4]] = [[23,34], [31,46]]
	assert np.allclose(rmatmul_result.data, np.array([[23, 34], [31, 46]]))


def test_matrix_iadd_isub_imul():
	m1 = Matrix([[1, 2], [3, 4]])
	m2 = Matrix([[5, 6], [7, 8]])

	result = m1.__iadd__(m2)
	assert result is m1
	assert np.allclose(m1.data, np.array([[6, 8], [10, 12]]))

	m1 = Matrix([[10, 15], [20, 25]])
	m2 = Matrix([[2, 3], [4, 5]])
	result = m1.__isub__(m2)
	assert result is m1
	assert np.allclose(m1.data, np.array([[8, 12], [16, 20]]))

	m1 = Matrix([[2, 3], [4, 5]])
	m2 = Matrix([[6, 7], [8, 9]])
	result = m1.__imul__(m2)
	assert result is m1
	assert np.allclose(m1.data, np.array([[12, 21], [32, 45]]))


def test_matrix_itruediv():
	m1 = Matrix([[10, 20], [30, 40]])
	m2 = Matrix([[2, 4], [5, 8]])

	result = m1.__itruediv__(m2)
	assert result is m1
	assert np.allclose(m1.data, np.array([[5.0, 5.0], [6.0, 5.0]]))


def test_matrix_imatmul():
	m1 = Matrix([[1, 2], [3, 4]])
	m2 = Matrix([[5, 6], [7, 8]])

	result = m1.__imatmul__(m2)
	assert result is m1
	assert np.allclose(m1.data, np.array([[19, 22], [43, 50]]))


# ---- matmul Matrix @ Vector ----

def test_matrix_matmul_vector_returns_vector():
	m = Matrix([[1, 0], [0, 1]])
	v = Vector([3, 4])
	result = m @ v
	assert isinstance(result, Vector)
	assert np.allclose(result.data, np.array([3.0, 4.0]))


def test_matrix_matmul_vector_rotation():
	theta = np.pi / 2
	rot = Matrix([[np.cos(theta), -np.sin(theta)],
	              [np.sin(theta),  np.cos(theta)]])
	v = Vector([1, 0])
	result = rot @ v
	assert isinstance(result, Vector)
	assert np.allclose(result.data, np.array([0.0, 1.0]), atol=1e-9)


def test_matrix_matmul_misaligned_vector_raises():
	m = Matrix([[1, 2, 3], [4, 5, 6]])
	v = Vector([1, 2])
	with pytest.raises(ValueError, match="not aligned for matrix multiplication"):
		_ = m @ v


# ---- solve ----

def test_matrix_solve_linear_system():
	a = Matrix([[2, 1], [5, 7]])
	b = Vector([11, 13])
	x = a.solve(b)
	assert isinstance(x, Vector)
	assert np.allclose(x.data, np.linalg.solve(np.array([[2, 1], [5, 7]]), np.array([11, 13])))


def test_matrix_solve_raises_for_non_square():
	m = Matrix([[1, 2, 3], [4, 5, 6]])
	b = Vector([1, 2])
	with pytest.raises(ValueError, match="square matrices"):
		m.solve(b)


# ---- full comparison operator coverage ----

def test_matrix_le_gt_ge_operators():
	m1 = Matrix([[1, 5], [3, 4]])
	m2 = Matrix([[2, 5], [1, 6]])

	le = m1 <= m2
	gt = m1 > m2
	ge = m1 >= m2

	assert isinstance(le, Matrix)
	assert isinstance(gt, Matrix)
	assert isinstance(ge, Matrix)
	assert np.array_equal(le.data, np.array([[True,  True],  [False, True]]))
	assert np.array_equal(gt.data, np.array([[False, False], [True,  False]]))
	assert np.array_equal(ge.data, np.array([[False, True],  [True,  False]]))


def test_matrix_ne_elementwise():
	m1 = Matrix([[1, 2], [3, 4]])
	m2 = Matrix([[1, 9], [3, 4]])
	ne = m1 != m2
	assert isinstance(ne, Matrix)
	assert np.array_equal(ne.data, np.array([[False, True], [False, False]]))


def test_matrix_comparison_with_wrong_type_returns_not_implemented():
	m = Matrix([[1, 2], [3, 4]])
	assert Matrix.__lt__(m, 42) is NotImplemented


# ---- scalar conversion ----

def test_matrix_float_raises_for_non_scalar_shape():
	m = Matrix([[1, 2], [3, 4]])
	with pytest.raises(ValueError, match="Cannot convert"):
		float(m)


def test_matrix_float_and_complex_for_single_element_matrix():
	m = Matrix([[7]])
	assert float(m) == 7.0
	assert complex(m) == 7 + 0j


# ---- shape / type guards ----

def test_matrix_sub_raises_for_shape_mismatch():
	m1 = Matrix([[1, 2], [3, 4]])
	m2 = Matrix([[1, 2, 3], [4, 5, 6]])
	with pytest.raises(ValueError, match="shapes must be the same"):
		_ = m1 - m2


def test_matrix_itruediv_raises_for_zero_denominator():
	m1 = Matrix([[4, 6], [8, 10]])
	m2 = Matrix([[2, 0], [4, 5]])
	with pytest.raises(ZeroDivisionError):
		m1 /= m2


def test_matrix_imatmul_raises_for_misaligned_shapes():
	m1 = Matrix([[1, 2], [3, 4]])
	m2 = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
	with pytest.raises(ValueError, match="not aligned for matrix multiplication"):
		m1 @= m2


# ---- utility methods ----

def test_matrix_astype_underlying_numpy_dtype():
	m = Matrix([[1, 2], [3, 4]])
	# The Matrix constructor always coerces to float64;
	# verify the raw numpy cast works correctly.
	raw = m.data.astype(np.int32)
	assert raw.dtype == np.dtype("int32")
	assert raw.tolist() == [[1, 2], [3, 4]]


def test_matrix_flatten_returns_flat_data():
	m = Matrix([[1, 2], [3, 4]])
	# flatten() yields a 1-D array; for Matrix it raises because the
	# constructor requires 2-D — verify the raw NumPy flatten is correct.
	flat = m.data.flatten()
	assert flat.shape == (4,)
	assert list(flat) == [1.0, 2.0, 3.0, 4.0]


def test_matrix_copy_is_independent():
	m = Matrix([[1, 2], [3, 4]])
	c = m.copy()
	c[0, 0] = 99
	assert m[0, 0] == 1


def test_matrix_deepcopy_is_independent():
	import copy
	m = Matrix([[1, 2], [3, 4]])
	c = copy.deepcopy(m)
	c[0, 0] = 99
	assert m[0, 0] == 1


def test_matrix_dtype_ndim_and_size_properties():
	m = Matrix([[1, 2], [3, 4]])
	assert m.dtype == np.dtype("float64")
	assert m.ndim == 2
	assert m.size == 4


# ---- determinant / trace / inverse cross-checks ----

def test_matrix_determinant_of_identity_is_one():
	m = Matrix([[1, 0], [0, 1]])
	assert np.isclose(m.determinant(), 1.0)


def test_matrix_trace_of_3x3():
	m = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
	assert np.isclose(m.trace(), 15.0)


def test_matrix_inverse_times_original_is_identity():
	m = Matrix([[2, 1], [5, 3]])
	identity = m @ m.inverse()
	assert np.allclose(identity.data, np.eye(2), atol=1e-9)
