from __future__ import annotations

import copy
from typing import Iterator

import numpy as np

from .array_base import Base, format_array
from ..utils.precision import a_tol, r_tol
from ..utils import isclose, clean

class Tensor(Base):
    """N-dimensional numeric array constrained to rank 3 or higher."""

    def __init__(self, data) -> None:
        super().__init__(data)
        if self.ndim < 3:
            raise ValueError("Tensor must be 3D or higher")

    def __str__(self) -> str:
        return format_array(self.data)

    def __repr__(self) -> str:
        return f"Tensor({format_array(self.data)})"

    def __bool__(self) -> bool:
        return self.size > 0

    def __float__(self):
        if self.size != 1:
            raise ValueError(f"Cannot convert Tensor with shape {self.shape} to float")
        return float(self.data.reshape(-1)[0])

    def __complex__(self):
        if self.size != 1:
            raise ValueError(f"Cannot convert Tensor with shape {self.shape} to complex")
        return complex(self.data.reshape(-1)[0])

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        array = self.data
        if dtype is not None:
            array = array.astype(dtype, copy=False)
        if copy is None:
            return array
        return np.array(array, copy=copy)

    def __format__(self, format_spec: str) -> str:
        formatted = np.array2string(
            self.data,
            formatter={
                "float_kind": lambda x: format(float(x), format_spec),
                "complex_kind": lambda x: format(complex(x), format_spec),
            },
        )
        return f"Tensor({formatted})"

    def __len__(self) -> int:
        return self.shape[0]

    def __iter__(self) -> Iterator[np.ndarray]:
        for i in range(self.shape[0]):
            yield self.data[i]

    def __contains__(self, item) -> bool:
        return bool(np.any(isclose(self.data, item)))

    def __setitem__(self, key, value) -> None:
        self.data[key] = value

    def __getitem__(self, key):
        result = self.data[key]
        if np.isscalar(result) or getattr(result, "ndim", 0) == 0:
            val = float(result)
            return int(val) if val.is_integer() else val
        if result.ndim >= 3:
            return Tensor(result)
        return result

    def __copy__(self) -> Tensor:
        return Tensor(self.data.copy())

    def __deepcopy__(self, memo) -> Tensor:
        return Tensor(copy.deepcopy(self.data, memo))

    def copy(self) -> Tensor:
        return self.__copy__()

    def any(self) -> bool:
        return self.data.any()

    def all(self) -> bool:
        return self.data.all()

    def reshape(self, *shape) -> Tensor:
        return Tensor(self.data.reshape(*shape))

    def flatten(self) -> np.ndarray:
        return self.data.flatten()

    def astype(self, dtype) -> Tensor:
        return Tensor(self.data.astype(dtype))

    def __eq__(self, other):
        if isinstance(other, Tensor):
            return Tensor(isclose(self.data, other.data))
        return NotImplemented

    def __ne__(self, other):
        if isinstance(other, Tensor):
            return Tensor(~isclose(self.data, other.data))
        return NotImplemented

    def __lt__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self.data < other.data)
        return NotImplemented

    def __le__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self.data <= other.data)
        return NotImplemented

    def __gt__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self.data > other.data)
        return NotImplemented

    def __ge__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self.data >= other.data)
        return NotImplemented

    def __neg__(self):
        return Tensor(-self.data)

    def __pos__(self):
        return Tensor(+self.data)

    def __abs__(self):
        return Tensor(np.abs(self.data))

    def __add__(self, other):
        if not isinstance(other, Tensor):
            return NotImplemented
        if self.shape != other.shape:
            raise ValueError("Tensor shapes must be the same for addition")
        result = np.empty(self.shape, dtype=np.float64)
        for idx in np.ndindex(self.shape):
            result[idx] = self.data[idx] + other.data[idx]
        return Tensor(result)

    def __sub__(self, other):
        if not isinstance(other, Tensor):
            return NotImplemented
        if self.shape != other.shape:
            raise ValueError("Tensor shapes must be the same for subtraction")
        result = np.empty(self.shape, dtype=np.float64)
        for idx in np.ndindex(self.shape):
            result[idx] = self.data[idx] - other.data[idx]
        return Tensor(result)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            result = np.empty(self.shape, dtype=np.float64)
            for idx in np.ndindex(self.shape):
                result[idx] = self.data[idx] * other
            return Tensor(result)
        if isinstance(other, Tensor):
            if self.shape != other.shape:
                raise ValueError("Tensor shapes must be the same for multiplication")
            result = np.empty(self.shape, dtype=np.float64)
            for idx in np.ndindex(self.shape):
                result[idx] = self.data[idx] * other.data[idx]
            return Tensor(result)
        return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            if isclose(other, 0.0):
                raise ZeroDivisionError("Cannot divide by zero")
            result = np.empty(self.shape, dtype=np.float64)
            for idx in np.ndindex(self.shape):
                result[idx] = clean(self.data[idx] / other)
            return Tensor(result)
        if isinstance(other, Tensor):
            if self.shape != other.shape:
                raise ValueError("Tensor shapes must be the same for division")
            if np.any(isclose(other.data, 0.0)):
                raise ZeroDivisionError("Cannot divide by zero")
            result = np.empty(self.shape, dtype=np.float64)
            for idx in np.ndindex(self.shape):
                result[idx] = clean(self.data[idx] / other.data[idx])
            return Tensor(result)
        return NotImplemented

    def __radd__(self, other):
        return self.__add__(other)

    def __rsub__(self, other):
        if not isinstance(other, Tensor):
            return NotImplemented
        if self.shape != other.shape:
            raise ValueError("Tensor shapes must be the same for subtraction")
        result = np.empty(self.shape, dtype=np.float64)
        for idx in np.ndindex(self.shape):
            result[idx] = other.data[idx] - self.data[idx]
        return Tensor(result)

    def __rmul__(self, other):
        if isinstance(other, (int, float)):
            result = np.empty(self.shape, dtype=np.float64)
            for idx in np.ndindex(self.shape):
                result[idx] = other * self.data[idx]
            return Tensor(result)
        if isinstance(other, Tensor):
            if self.shape != other.shape:
                raise ValueError("Tensor shapes must be the same for multiplication")
            result = np.empty(self.shape, dtype=np.float64)
            for idx in np.ndindex(self.shape):
                result[idx] = other.data[idx] * self.data[idx]
            return Tensor(result)
        return NotImplemented

    def __rtruediv__(self, other):
        if isinstance(other, (int, float)):
            if np.any(isclose(self.data, 0.0)):
                raise ZeroDivisionError("Cannot divide by zero")
            result = np.empty(self.shape, dtype=np.float64)
            for idx in np.ndindex(self.shape):
                result[idx] = clean(other / self.data[idx])
            return Tensor(result)
        if isinstance(other, Tensor):
            if self.shape != other.shape:
                raise ValueError("Tensor shapes must be the same for division")
            if np.any(isclose(self.data, 0.0)):
                raise ZeroDivisionError("Cannot divide by zero")
            result = np.empty(self.shape, dtype=np.float64)
            for idx in np.ndindex(self.shape):
                result[idx] = clean(other.data[idx] / self.data[idx])
            return Tensor(result)
        return NotImplemented

    def __iadd__(self, other):
        if not isinstance(other, Tensor):
            return NotImplemented
        if self.shape != other.shape:
            raise ValueError("Tensor shapes must be the same for addition")
        for idx in np.ndindex(self.shape):
            self.data[idx] += other.data[idx]
        return self

    def __isub__(self, other):
        if not isinstance(other, Tensor):
            return NotImplemented
        if self.shape != other.shape:
            raise ValueError("Tensor shapes must be the same for subtraction")
        for idx in np.ndindex(self.shape):
            self.data[idx] -= other.data[idx]
        return self

    def __imul__(self, other):
        if isinstance(other, (int, float)):
            for idx in np.ndindex(self.shape):
                self.data[idx] *= other
            return self
        if isinstance(other, Tensor):
            if self.shape != other.shape:
                raise ValueError("Tensor shapes must be the same for multiplication")
            for idx in np.ndindex(self.shape):
                self.data[idx] *= other.data[idx]
            return self
        return NotImplemented

    def __itruediv__(self, other):
        if isinstance(other, (int, float)):
            if np.isclose(other, 0.0, rtol=r_tol, atol=a_tol):
                raise ZeroDivisionError("Cannot divide by zero")
            for idx in np.ndindex(self.shape):
                self.data[idx] /= other
            return self
        if isinstance(other, Tensor):
            if self.shape != other.shape:
                raise ValueError("Tensor shapes must be the same for division")
            if np.any(isclose(other.data, 0.0)):
                raise ZeroDivisionError("Cannot divide by zero")
            for idx in np.ndindex(self.shape):
                self.data[idx] = clean(self.data[idx] / other.data[idx])
            return self
        return NotImplemented

    def contract(self, axes) -> Tensor:
        return Tensor(np.tensordot(self.data, self.data, axes=axes))

    def transpose(self, axes=None) -> Tensor:
        return Tensor(np.transpose(self.data, axes=axes))

    def einsum(self, subscripts, *operands) -> Tensor:
        return Tensor(np.einsum(subscripts, self.data, *operands))
