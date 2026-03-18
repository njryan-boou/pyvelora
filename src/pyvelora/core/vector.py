from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any, Iterator, Literal

import numpy as np

from .array_base import Base, format_array
from ..utils import allclose, isclose, zero_threshold, clean
from ..utils.precision import a_tol, r_tol

if TYPE_CHECKING:
    from .matrix import Matrix

CoordinateType = Literal["polar", "spherical", "cylindrical", "cartesian"]


class Vector(Base):
    """One-dimensional numeric array with vector-specific operations."""

    def __init__(
        self, 
        data: Any, 
        type: CoordinateType | None = None, 
        degrees: bool = True,
    ) -> None:
        super().__init__(data)

        if self.ndim != 1:
            raise ValueError("Vector must be 1D")

        if type == "polar":
            if self.size != 2:
                raise ValueError("Polar coordinates must be a 1D vector of size 2 (r, theta)")
            r, theta = self.data
            if degrees:
                theta = np.deg2rad(theta)
            self.data = np.array([r * np.cos(theta), r * np.sin(theta)])

        elif type == "spherical":
            if self.size != 3:
                raise ValueError("Spherical coordinates must be a 1D vector of size 3 (r, theta, phi)")
            r, theta, phi = self.data
            if degrees:
                theta = np.deg2rad(theta)
                phi = np.deg2rad(phi)
            self.data = np.array(
                [
                    r * np.sin(theta) * np.cos(phi),
                    r * np.sin(theta) * np.sin(phi),
                    r * np.cos(theta),
                ]
            )

        elif type == "cylindrical":
            if self.size != 3:
                raise ValueError("Cylindrical coordinates must be a 1D vector of size 3 (rho, phi, z)")
            rho, phi, z = self.data
            if degrees:
                phi = np.deg2rad(phi)
            self.data = np.array(
                [
                    rho * np.cos(phi),
                    rho * np.sin(phi),
                    z,
                ]
            )

        elif type not in (None,):
            raise ValueError(f"Unknown coordinate type: {type}")

    def __str__(self) -> str:
        return format_array(self.data)

    def __repr__(self) -> str:
        return f"Vector({format_array(self.data)})"

    def __bool__(self) -> bool:
        return self.size > 0

    def __float__(self):
        if self.size != 1:
            raise ValueError(f"Cannot convert Vector with shape {self.shape} to float")
        return float(self.data.reshape(-1)[0])

    def __complex__(self):
        if self.size != 1:
            raise ValueError(f"Cannot convert Vector with shape {self.shape} to complex")
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
        return f"Vector({formatted})"

    def __len__(self) -> int:
        return self.size

    def __iter__(self) -> Iterator[int | float]:
        for x in self.data:
            val = float(x)
            yield int(val) if val.is_integer() else val

    def __contains__(self, item) -> bool:
        return bool(np.any(isclose(self.data, item)))

    def __setitem__(self, key, value) -> None:
        self.data[key] = value

    def __getitem__(self, key):
        result = self.data[key]
        if np.isscalar(result) or getattr(result, "ndim", 0) == 0:
            val = float(result)
            return int(val) if val.is_integer() else val
        return Vector(result)

    def __copy__(self) -> Vector:
        return Vector(self.data.copy())

    def __deepcopy__(self, memo) -> Vector:
        return Vector(copy.deepcopy(self.data, memo))

    def copy(self) -> Vector:
        return self.__copy__()

    def magnitude(self) -> float:
        return np.sqrt(np.sum(self.data ** 2))

    def normalize(self) -> Vector:
        mag = self.magnitude()
        if mag == 0:
            raise ValueError("Cannot normalize a zero vector")
        return Vector(self.data / mag)

    def any(self) -> bool:
        return self.data.any()

    def all(self) -> bool:
        return self.data.all()

    def flatten(self) -> Vector:
        return Vector(self.data.flatten())

    def __add__(self, other):
        if not isinstance(other, Vector):
            return NotImplemented
        if self.shape != other.shape:
            raise ValueError("Vector shapes must be the same for addition")
        result = [clean(a + b) for a, b in zip(self.data, other.data)]
        return Vector(result)

    def __sub__(self, other):
        if not isinstance(other, Vector):
            return NotImplemented
        if self.shape != other.shape:
            raise ValueError("Vector shapes must be the same for subtraction")
        result = [clean(a - b) for a, b in zip(self.data, other.data)]
        return Vector(result)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            result = np.empty(self.size, dtype=np.float64)
            for i in range(self.size):
                result[i] = clean(self.data[i] * other)
            return Vector(result)
        if isinstance(other, Vector):
            if self.shape != other.shape:
                raise ValueError("Vector shapes must be the same for multiplication")
            result = np.empty(self.size, dtype=np.float64)
            for i in range(self.size):
                result[i] = clean(self.data[i] * other.data[i])
            return Vector(result)
        return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            if np.isclose(other, 0.0, rtol=r_tol, atol=a_tol):
                raise ZeroDivisionError("Cannot divide by zero")
            result = np.empty(self.size, dtype=np.float64)
            for i in range(self.size):
                result[i] = clean(self.data[i] / other)
            return Vector(result)
        if isinstance(other, Vector):
            if self.shape != other.shape:
                raise ValueError("Vector shapes must be the same for division")
            if np.any(isclose(other.data, 0.0)):
                raise ZeroDivisionError("Cannot divide by zero")
            result = np.empty(self.size, dtype=np.float64)
            for i in range(self.size):
                result[i] = clean(self.data[i] / other.data[i])
            return Vector(result)
        return NotImplemented

    def __matmul__(self, other):
        if isinstance(other, Vector):
            if self.shape != other.shape:
                raise ValueError("Vector shapes are not aligned for matrix multiplication")
            total = 0.0
            for i in range(self.size):
                total += clean(self.data[i] * other.data[i])
            return float(total)

        from .matrix import Matrix
        if isinstance(other, Matrix):
            if self.shape[0] != other.shape[0]:
                raise ValueError("Vector and Matrix shapes are not aligned for matrix multiplication")
            cols = other.shape[1]
            out = np.empty(cols, dtype=np.float64)
            for j in range(cols):
                total = 0.0
                for i in range(self.size):
                    total += clean(self.data[i] * other.data[i, j])
                out[j] = total
            return Vector(out)
        return NotImplemented

    def __radd__(self, other):
        return self.__add__(other)

    def __rsub__(self, other):
        if not isinstance(other, Vector):
            return NotImplemented
        if self.shape != other.shape:
            raise ValueError("Vector shapes must be the same for subtraction")
        result = [clean(b - a) for a, b in zip(self.data, other.data)]
        return Vector(result)

    def __rmul__(self, other):
        if isinstance(other, (int, float)):
            result = np.empty(self.size, dtype=np.float64)
            for i in range(self.size):
                result[i] = other * self.data[i]
            return Vector(result)
        if isinstance(other, Vector):
            if self.shape != other.shape:
                raise ValueError("Vector shapes must be the same for multiplication")
            result = np.empty(self.size, dtype=np.float64)
            for i in range(self.size):
                result[i] = other.data[i] * self.data[i]
            return Vector(result)
        return NotImplemented

    def __rtruediv__(self, other):
        if isinstance(other, (int, float)):
            if np.any(isclose(self.data, 0.0)):
                raise ZeroDivisionError("Cannot divide by zero")
            result = np.empty(self.size, dtype=np.float64)
            for i in range(self.size):
                result[i] = other / self.data[i]
            return Vector(result)
        if isinstance(other, Vector):
            if self.shape != other.shape:
                raise ValueError("Vector shapes must be the same for division")
            if np.any(isclose(self.data, 0.0)):
                raise ZeroDivisionError("Cannot divide by zero")
            result = np.empty(self.size, dtype=np.float64)
            for i in range(self.size):
                result[i] = other.data[i] / self.data[i]
            return Vector(result)
        return NotImplemented

    def __rmatmul__(self, other):
        if isinstance(other, Vector):
            if self.shape != other.shape:
                raise ValueError("Vector shapes are not aligned for matrix multiplication")
            total = 0.0
            for i in range(self.size):
                total += other.data[i] * self.data[i]
            return float(total)

        from .matrix import Matrix
        if isinstance(other, Matrix):
            if other.shape[1] != self.shape[0]:
                raise ValueError("Matrix and Vector shapes are not aligned for matrix multiplication")
            rows = other.shape[0]
            out = np.empty(rows, dtype=np.float64)
            for i in range(rows):
                total = 0.0
                for j in range(self.size):
                    total += other.data[i, j] * self.data[j]
                out[i] = total
            return Vector(out)
        return NotImplemented

    def __iadd__(self, other):
        if not isinstance(other, Vector):
            return NotImplemented
        if self.shape != other.shape:
            raise ValueError("Vector shapes must be the same for addition")
        for i in range(self.size):
            self.data[i] += other.data[i]
        return self

    def __isub__(self, other):
        if not isinstance(other, Vector):
            return NotImplemented
        if self.shape != other.shape:
            raise ValueError("Vector shapes must be the same for subtraction")
        for i in range(self.size):
            self.data[i] -= other.data[i]
        return self

    def __imul__(self, other):
        if isinstance(other, (int, float)):
            for i in range(self.size):
                self.data[i] *= other
            return self
        if isinstance(other, Vector):
            if self.shape != other.shape:
                raise ValueError("Vector shapes must be the same for multiplication")
            for i in range(self.size):
                self.data[i] *= other.data[i]
            return self
        return NotImplemented

    def __itruediv__(self, other):
        if isinstance(other, (int, float)):
            if isclose(other, 0.0):
                raise ZeroDivisionError("Cannot divide by zero")
            for i in range(self.size):
                self.data[i] /= other
            return self
        if isinstance(other, Vector):
            if self.shape != other.shape:
                raise ValueError("Vector shapes must be the same for division")
            if np.any(isclose(other.data, 0.0)):
                raise ZeroDivisionError("Cannot divide by zero")
            for i in range(self.size):
                self.data[i] /= other.data[i]
            return self
        return NotImplemented

    def __imatmul__(self, other):
        if not isinstance(other, Vector):
            return NotImplemented
        if self.shape != other.shape:
            raise ValueError("Vector shapes are not aligned for matrix multiplication")
        total = 0.0
        for i in range(self.size):
            total += self.data[i] * other.data[i]
        self.data = np.array([total], dtype=np.float64)
        return self

    def __neg__(self):
        return Vector(-self.data)

    def __pos__(self):
        return Vector(+self.data)

    def __abs__(self):
        return Vector(np.abs(self.data))

    def __invert__(self):
        return Vector(~self.data)

    def __round__(self, n=None):
        return Vector(np.round(self.data, n))

    def __eq__(self, other):
        if isinstance(other, Vector):
            return Vector(isclose(self.data, other.data))
        return NotImplemented

    def __ne__(self, other):
        if isinstance(other, Vector):
            return Vector(~isclose(self.data, other.data))
        return NotImplemented

    def __lt__(self, other):
        if isinstance(other, Vector):
            return Vector(self.data < other.data)
        return NotImplemented

    def __le__(self, other):
        if isinstance(other, Vector):
            return Vector(self.data <= other.data)
        return NotImplemented

    def __gt__(self, other):
        if isinstance(other, Vector):
            return Vector(self.data > other.data)
        return NotImplemented

    def __ge__(self, other):
        if isinstance(other, Vector):
            return Vector(self.data >= other.data)
        return NotImplemented
