from __future__ import annotations

from pyvelora.core.array_base import Base, format_array
from pyvelora.core.tensor.indexing import TensorIndexing
from pyvelora.core.tensor.arithmetic import TensorArithmetic
from pyvelora.core.tensor.comparison import TensorComparison


class Tensor(Base, TensorIndexing, TensorArithmetic, TensorComparison):
    """N-dimensional numeric array constrained to rank 3 or higher."""

    def __init__(self, data: list) -> None:
        super().__init__(data=data)
        if self.ndim < 3:
            raise ValueError("Tensor must be 3D or higher")

    def __str__(self) -> str:
        return format_array(self.data)

    def __repr__(self) -> str:
        return f"Tensor({format_array(self.data)})"

    def transpose(self, axes=None) -> Tensor:
        if axes is None:
            axes = tuple(reversed(range(self.ndim)))
        source_shape = self.shape
        result_shape = tuple(source_shape[axis] for axis in axes)

        total_size = 1
        for dimension in source_shape:
            total_size *= dimension

        flat_result = [0.0 for _ in range(total_size)]
        result = flat_result[0]
        for dimension in reversed(result_shape):
            grouped = []
            for index in range(0, len(flat_result), dimension):
                grouped.append(flat_result[index:index + dimension])
            flat_result = grouped
        result = flat_result[0] if result_shape else result

        source_indices = [()]
        for dimension in source_shape:
            next_indices = []
            for index_prefix in source_indices:
                for index in range(dimension):
                    next_indices.append(index_prefix + (index,))
            source_indices = next_indices

        for source_index in source_indices:
            target_index = tuple(source_index[axis] for axis in axes)

            value = self.data
            for part in source_index:
                value = value[part]

            target = result
            for part in target_index[:-1]:
                target = target[part]
            target[target_index[-1]] = value

        return type(self)(result)

    def contract(self, other: Tensor | None = None, axes=1) -> Tensor:
        if other is None:
            other = self
        if not isinstance(other, type(self)):
            return NotImplemented

        left_shape = self.shape
        right_shape = other.shape
        if isinstance(axes, int):
            left_axes = list(range(len(left_shape) - axes, len(left_shape)))
            right_axes = list(range(axes))
        else:
            left_axes, right_axes = axes
            if isinstance(left_axes, int):
                left_axes = [left_axes]
            else:
                left_axes = list(left_axes)
            if isinstance(right_axes, int):
                right_axes = [right_axes]
            else:
                right_axes = list(right_axes)

        left_free = [axis for axis in range(len(left_shape)) if axis not in left_axes]
        right_free = [axis for axis in range(len(right_shape)) if axis not in right_axes]
        contract_shape = tuple(left_shape[axis] for axis in left_axes)
        result_shape = tuple(left_shape[axis] for axis in left_free) + tuple(right_shape[axis] for axis in right_free)

        left_free_indices = [()]
        for dimension in tuple(left_shape[axis] for axis in left_free):
            next_indices = []
            for index_prefix in left_free_indices:
                for index in range(dimension):
                    next_indices.append(index_prefix + (index,))
            left_free_indices = next_indices

        right_free_indices = [()]
        for dimension in tuple(right_shape[axis] for axis in right_free):
            next_indices = []
            for index_prefix in right_free_indices:
                for index in range(dimension):
                    next_indices.append(index_prefix + (index,))
            right_free_indices = next_indices

        contract_indices = [()]
        for dimension in contract_shape:
            next_indices = []
            for index_prefix in contract_indices:
                for index in range(dimension):
                    next_indices.append(index_prefix + (index,))
            contract_indices = next_indices

        result_values = []
        for left_index_free in left_free_indices:
            for right_index_free in right_free_indices:
                total = 0.0
                for contract_index in contract_indices:
                    left_index = []
                    right_index = []
                    left_free_position = 0
                    right_free_position = 0
                    contract_position = 0

                    for axis in range(len(left_shape)):
                        if axis in left_axes:
                            left_index.append(contract_index[contract_position])
                            contract_position += 1
                        else:
                            left_index.append(left_index_free[left_free_position])
                            left_free_position += 1

                    contract_position = 0
                    for axis in range(len(right_shape)):
                        if axis in right_axes:
                            right_index.append(contract_index[contract_position])
                            contract_position += 1
                        else:
                            right_index.append(right_index_free[right_free_position])
                            right_free_position += 1

                    left_value = self.data
                    for part in left_index:
                        left_value = left_value[part]

                    right_value = other.data
                    for part in right_index:
                        right_value = right_value[part]

                    total += left_value * right_value

                result_values.append(total)

        if not result_shape:
            result = result_values[0]
        else:
            nested = result_values[:]
            for dimension in reversed(result_shape):
                grouped = []
                for index in range(0, len(nested), dimension):
                    grouped.append(nested[index:index + dimension])
                nested = grouped
            result = nested[0]

        result_depth = 0
        current = result
        while isinstance(current, list):
            result_depth += 1
            if not current:
                break
            current = current[0]

        if result_depth < 3:
            raise ValueError("Tensor must be 3D or higher")
        return type(self)(result)

    def einsum(self, subscripts: str) -> Tensor:
        if subscripts != "ijk->ijk":
            raise ValueError("Tensor must be 3D or higher")

        result = []
        stack = [(self.data, result)]
        while stack:
            source, target = stack.pop()
            for item in source:
                if isinstance(item, list):
                    nested = []
                    target.append(nested)
                    stack.append((item, nested))
                else:
                    target.append(item)
        return type(self)(result)

    def outer(self, other: Tensor) -> Tensor:
        if not isinstance(other, type(self)):
            return NotImplemented

        left_shape = self.shape
        right_shape = other.shape
        left_indices = [()]
        for dimension in left_shape:
            next_indices = []
            for index_prefix in left_indices:
                for index in range(dimension):
                    next_indices.append(index_prefix + (index,))
            left_indices = next_indices

        right_indices = [()]
        for dimension in right_shape:
            next_indices = []
            for index_prefix in right_indices:
                for index in range(dimension):
                    next_indices.append(index_prefix + (index,))
            right_indices = next_indices

        result_values = []
        for left_index in left_indices:
            left_value = self.data
            for part in left_index:
                left_value = left_value[part]
            for right_index in right_indices:
                right_value = other.data
                for part in right_index:
                    right_value = right_value[part]
                result_values.append(left_value * right_value)

        result_shape = left_shape + right_shape
        nested = result_values[:]
        for dimension in reversed(result_shape):
            grouped = []
            for index in range(0, len(nested), dimension):
                grouped.append(nested[index:index + dimension])
            nested = grouped
        return type(self)(nested[0])

    def inner(self, other: Tensor) -> Tensor:
        if not isinstance(other, type(self)):
            return NotImplemented

        left_shape = self.shape
        right_shape = other.shape
        left_free = [axis for axis in range(len(left_shape) - 1)]
        right_free = [axis for axis in range(len(right_shape) - 1)]
        contract_size = left_shape[-1]
        result_shape = tuple(left_shape[axis] for axis in left_free) + tuple(right_shape[axis] for axis in right_free)

        left_free_indices = [()]
        for dimension in tuple(left_shape[axis] for axis in left_free):
            next_indices = []
            for index_prefix in left_free_indices:
                for index in range(dimension):
                    next_indices.append(index_prefix + (index,))
            left_free_indices = next_indices

        right_free_indices = [()]
        for dimension in tuple(right_shape[axis] for axis in right_free):
            next_indices = []
            for index_prefix in right_free_indices:
                for index in range(dimension):
                    next_indices.append(index_prefix + (index,))
            right_free_indices = next_indices

        result_values = []
        for left_index_free in left_free_indices:
            for right_index_free in right_free_indices:
                total = 0.0
                for contract_index in range(contract_size):
                    left_value = self.data
                    for part in left_index_free + (contract_index,):
                        left_value = left_value[part]

                    right_value = other.data
                    for part in right_index_free + (contract_index,):
                        right_value = right_value[part]

                    total += left_value * right_value
                result_values.append(total)

        nested = result_values[:]
        for dimension in reversed(result_shape):
            grouped = []
            for index in range(0, len(nested), dimension):
                grouped.append(nested[index:index + dimension])
            nested = grouped
        return type(self)(nested[0])

    def tensordot(self, other: Tensor, axes=1) -> Tensor:
        if not isinstance(other, type(self)):
            return NotImplemented

        left_shape = self.shape
        right_shape = other.shape
        if isinstance(axes, int):
            left_axes = list(range(len(left_shape) - axes, len(left_shape)))
            right_axes = list(range(axes))
        else:
            left_axes, right_axes = axes
            if isinstance(left_axes, int):
                left_axes = [left_axes]
            else:
                left_axes = list(left_axes)
            if isinstance(right_axes, int):
                right_axes = [right_axes]
            else:
                right_axes = list(right_axes)

        left_free = [axis for axis in range(len(left_shape)) if axis not in left_axes]
        right_free = [axis for axis in range(len(right_shape)) if axis not in right_axes]
        contract_shape = tuple(left_shape[axis] for axis in left_axes)
        result_shape = tuple(left_shape[axis] for axis in left_free) + tuple(right_shape[axis] for axis in right_free)

        left_free_indices = [()]
        for dimension in tuple(left_shape[axis] for axis in left_free):
            next_indices = []
            for index_prefix in left_free_indices:
                for index in range(dimension):
                    next_indices.append(index_prefix + (index,))
            left_free_indices = next_indices

        right_free_indices = [()]
        for dimension in tuple(right_shape[axis] for axis in right_free):
            next_indices = []
            for index_prefix in right_free_indices:
                for index in range(dimension):
                    next_indices.append(index_prefix + (index,))
            right_free_indices = next_indices

        contract_indices = [()]
        for dimension in contract_shape:
            next_indices = []
            for index_prefix in contract_indices:
                for index in range(dimension):
                    next_indices.append(index_prefix + (index,))
            contract_indices = next_indices

        result_values = []
        for left_index_free in left_free_indices:
            for right_index_free in right_free_indices:
                total = 0.0
                for contract_index in contract_indices:
                    left_index = []
                    right_index = []
                    left_free_position = 0
                    right_free_position = 0
                    contract_position = 0

                    for axis in range(len(left_shape)):
                        if axis in left_axes:
                            left_index.append(contract_index[contract_position])
                            contract_position += 1
                        else:
                            left_index.append(left_index_free[left_free_position])
                            left_free_position += 1

                    contract_position = 0
                    for axis in range(len(right_shape)):
                        if axis in right_axes:
                            right_index.append(contract_index[contract_position])
                            contract_position += 1
                        else:
                            right_index.append(right_index_free[right_free_position])
                            right_free_position += 1

                    left_value = self.data
                    for part in left_index:
                        left_value = left_value[part]

                    right_value = other.data
                    for part in right_index:
                        right_value = right_value[part]

                    total += left_value * right_value
                result_values.append(total)

        if not result_shape:
            raise ValueError("Tensor must be 3D or higher")

        nested = result_values[:]
        for dimension in reversed(result_shape):
            grouped = []
            for index in range(0, len(nested), dimension):
                grouped.append(nested[index:index + dimension])
            nested = grouped
        return type(self)(nested[0])

    def kron(self, other: Tensor) -> Tensor:
        if not isinstance(other, type(self)):
            return NotImplemented

        left_shape = self.shape
        right_shape = other.shape
        ndim = len(left_shape)
        result_shape = tuple(left_shape[i] * right_shape[i] for i in range(ndim))

        # Compute result strides (row-major)
        result_strides = [1] * ndim
        for i in range(ndim - 2, -1, -1):
            result_strides[i] = result_strides[i + 1] * result_shape[i + 1]

        total = 1
        for d in result_shape:
            total *= d
        result_values = [0.0] * total

        # Iterate all left indices
        left_indices = [()]
        for d in left_shape:
            next_i = []
            for prefix in left_indices:
                for j in range(d):
                    next_i.append(prefix + (j,))
            left_indices = next_i

        # Iterate all right indices
        right_indices = [()]
        for d in right_shape:
            next_i = []
            for prefix in right_indices:
                for j in range(d):
                    next_i.append(prefix + (j,))
            right_indices = next_i

        for left_idx in left_indices:
            left_value = self.data
            for part in left_idx:
                left_value = left_value[part]
            for right_idx in right_indices:
                right_value = other.data
                for part in right_idx:
                    right_value = right_value[part]
                flat_idx = sum(
                    (left_idx[i] * right_shape[i] + right_idx[i]) * result_strides[i]
                    for i in range(ndim)
                )
                result_values[flat_idx] = left_value * right_value

        nested = result_values[:]
        for dimension in reversed(result_shape):
            grouped = []
            for index in range(0, len(nested), dimension):
                grouped.append(nested[index:index + dimension])
            nested = grouped
        return type(self)(nested[0])
