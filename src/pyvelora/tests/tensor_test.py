import copy

import numpy as np
import pytest

from pyvelora import Tensor


def test_tensor_init_requires_3d_or_higher():
    with pytest.raises(ValueError, match="Tensor must be 3D or higher"):
        Tensor([[1, 2], [3, 4]])


def test_tensor_properties_and_repr_str():
    t = Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

    assert t.shape == (2, 2, 2)
    assert t.ndim == 3
    assert t.size == 8
    assert t.dtype == np.dtype("float64")
    assert repr(t).startswith("Tensor(")
    assert "[[[1 2]" in str(t)


def test_tensor_getitem_scalar_and_slice():
    t = Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

    assert t[0, 0, 1] == 2
    sub = t[0]
    assert isinstance(sub, np.ndarray)
    assert sub.shape == (2, 2)


def test_tensor_setitem_and_contains():
    t = Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

    t[0, 0, 0] = 9

    assert t[0, 0, 0] == 9
    assert 9 in t
    assert 10 not in t


def test_tensor_copy_and_deepcopy_are_independent():
    t = Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

    c1 = t.copy()
    c2 = copy.deepcopy(t)
    c1[0, 0, 0] = 99
    c2[0, 0, 1] = 88

    assert t[0, 0, 0] == 1
    assert t[0, 0, 1] == 2


def test_tensor_add_sub_mul_div_with_tensor_and_scalar():
    t1 = Tensor([[[2, 4], [6, 8]], [[10, 12], [14, 16]]])
    t2 = Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

    assert np.allclose((t1 + t2).data, np.array([[[3, 6], [9, 12]], [[15, 18], [21, 24]]]))
    assert np.allclose((t1 - t2).data, np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]))
    assert np.allclose((t2 * 2).data, np.array([[[2, 4], [6, 8]], [[10, 12], [14, 16]]]))
    assert np.allclose((t1 / t2).data, np.array([[[2, 2], [2, 2]], [[2, 2], [2, 2]]]))


def test_tensor_division_by_zero_raises():
    t = Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    z = Tensor([[[1, 0], [1, 1]], [[1, 1], [1, 1]]])

    with pytest.raises(ZeroDivisionError, match="Cannot divide by zero"):
        _ = t / z


def test_tensor_shape_mismatch_raises_for_binary_ops():
    t1 = Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    t2 = Tensor([[[1, 2, 3], [4, 5, 6]]])

    with pytest.raises(ValueError, match="shapes must be the same"):
        _ = t1 + t2


def test_tensor_comparisons_return_tensor():
    t1 = Tensor([[[1, 5], [3, 4]], [[1, 1], [9, 0]]])
    t2 = Tensor([[[2, 5], [1, 6]], [[1, 3], [9, 1]]])

    lt = t1 < t2
    eq = t1 == t1.copy()

    assert isinstance(lt, Tensor)
    assert isinstance(eq, Tensor)
    assert lt.data.shape == t1.shape
    assert eq.all() == True


def test_tensor_inplace_ops_mutate_instance():
    t = Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    other = Tensor([[[1, 1], [1, 1]], [[1, 1], [1, 1]]])

    original_id = id(t)
    t += other
    t *= 2
    t /= 2

    assert id(t) == original_id
    assert np.allclose(t.data, np.array([[[2, 3], [4, 5]], [[6, 7], [8, 9]]]))


def test_tensor_reshape_transpose_contract_and_einsum():
    t = Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

    reshaped = t.reshape(4, 2, 1)
    transposed = t.transpose((1, 0, 2))
    contracted = t.contract(axes=1)
    einsummed = t.einsum("ijk->ijk")

    assert isinstance(reshaped, Tensor)
    assert reshaped.shape == (4, 2, 1)
    assert isinstance(transposed, Tensor)
    assert transposed.shape == (2, 2, 2)
    assert isinstance(contracted, Tensor)
    assert contracted.shape == (2, 2, 2, 2)
    assert np.array_equal(contracted.data, np.tensordot(t.data, t.data, axes=1))
    assert isinstance(einsummed, Tensor)
    assert np.array_equal(einsummed.data, np.einsum("ijk->ijk", t.data))


def test_tensor_flatten_returns_numpy_array():
    t = Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

    flat = t.flatten()

    assert isinstance(flat, np.ndarray)
    assert flat.shape == (8,)
    assert np.array_equal(flat, np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=float))


def test_tensor_bool_len_iter_and_any_all():
    t = Tensor([[[1, 0], [3, 4]], [[5, 6], [7, 8]]])

    assert bool(t) is True
    assert len(t) == 2
    chunks = list(t)
    assert len(chunks) == 2
    assert chunks[0].shape == (2, 2)
    assert bool(t.any()) is True
    assert bool(t.all()) is False


def test_tensor_bool_false_for_zero_size_tensor():
    t = Tensor(np.zeros((1, 1, 0)))
    assert bool(t) is False


def test_tensor_float_and_complex_for_single_element_tensor():
    t = Tensor([[[1]]])
    assert float(t) == 1.0
    assert complex(t) == 1 + 0j


def test_tensor_float_raises_for_non_scalar_tensor():
    t = Tensor([[[1, 2]]])
    with pytest.raises(ValueError, match="Cannot convert Tensor"):
        float(t)


def test_tensor_array_and_format_methods():
    t = Tensor([[[1.5, 2.5], [3.5, 4.5]], [[5.5, 6.5], [7.5, 8.5]]])

    raw = np.array(t)
    casted = t.__array__(dtype=np.int32)
    formatted = format(t, ".1f")

    assert np.array_equal(raw, t.data)
    assert casted.dtype == np.dtype("int32")
    assert formatted.startswith("Tensor(")


def test_tensor_getitem_returns_tensor_for_3d_slice():
    t = Tensor(np.arange(27).reshape(3, 3, 3))

    result = t[:, :, :2]

    assert isinstance(result, Tensor)
    assert result.shape == (3, 3, 2)


def test_tensor_unary_ops():
    t = Tensor([[[-1, 2], [-3, 4]], [[-5, 6], [-7, 8]]])

    assert np.array_equal((-t).data, np.array([[[1, -2], [3, -4]], [[5, -6], [7, -8]]]))
    assert np.array_equal((+t).data, t.data)
    assert np.array_equal(abs(t).data, np.abs(t.data))


def test_tensor_reverse_ops_with_scalar_and_tensor():
    t = Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    o = Tensor([[[2, 3], [4, 5]], [[6, 7], [8, 9]]])

    assert np.array_equal((2 * t).data, 2 * t.data)
    assert np.array_equal((o - t).data, o.data - t.data)
    assert np.array_equal((o / t).data, o.data / t.data)
    assert np.array_equal((2 / t).data, 2 / t.data)


def test_tensor_inplace_shape_mismatch_raises():
    t1 = Tensor(np.ones((2, 2, 2)))
    t2 = Tensor(np.ones((1, 2, 3)))

    with pytest.raises(ValueError, match="shapes must be the same"):
        t1 += t2


def test_tensor_scalar_division_by_zero_raises():
    t = Tensor(np.ones((2, 2, 2)))

    with pytest.raises(ZeroDivisionError, match="Cannot divide by zero"):
        _ = t / 0

    with pytest.raises(ZeroDivisionError, match="Cannot divide by zero"):
        _ = 1 / Tensor(np.array([[[1, 0], [1, 1]], [[1, 1], [1, 1]]]))


def test_tensor_wrong_type_operations_return_not_implemented():
    t = Tensor(np.ones((2, 2, 2)))

    assert t.__add__(object()) is NotImplemented
    assert t.__sub__(object()) is NotImplemented
    assert t.__mul__(object()) is NotImplemented
    assert t.__truediv__(object()) is NotImplemented
    assert t.__lt__(object()) is NotImplemented
    assert t.__le__(object()) is NotImplemented
    assert t.__gt__(object()) is NotImplemented
    assert t.__ge__(object()) is NotImplemented


def test_tensor_astype_preserves_tensor_type_and_values():
    t = Tensor(np.arange(8).reshape(2, 2, 2))
    casted = t.astype(np.int32)

    assert isinstance(casted, Tensor)
    assert casted.dtype == np.dtype("float64")
    assert np.array_equal(casted.data, t.data)


def test_tensor_copy_via_copy_module_is_independent():
    t = Tensor(np.arange(8).reshape(2, 2, 2))
    c = copy.copy(t)
    c[0, 0, 0] = 99

    assert t[0, 0, 0] == 0


def test_tensor_transpose_default_matches_numpy():
    t = Tensor(np.arange(24).reshape(2, 3, 4))
    transposed = t.transpose()

    assert isinstance(transposed, Tensor)
    assert np.array_equal(transposed.data, np.transpose(t.data))


def test_tensor_contract_or_einsum_reducing_below_3d_raises():
    t = Tensor(np.arange(8).reshape(2, 2, 2))

    with pytest.raises(ValueError, match="Tensor must be 3D or higher"):
        t.contract(axes=([0, 1, 2], [0, 1, 2]))

    with pytest.raises(ValueError, match="Tensor must be 3D or higher"):
        t.einsum("ijk->ik")


def test_tensor_reshape_to_less_than_3d_raises():
    t = Tensor(np.arange(8).reshape(2, 2, 2))

    with pytest.raises(ValueError, match="Tensor must be 3D or higher"):
        t.reshape(4, 2)


def test_tensor_contains_uses_tolerance():
    t = Tensor(np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]))
    assert 1.0 + 5e-13 in t
