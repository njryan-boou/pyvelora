from __future__ import annotations

import copy
from typing import Iterator

import numpy as np

from .array_base import Base, format_array
from .vector import Vector
from ..utils.precision import a_tol, r_tol
from ..utils import isclose, clean

class Matrix(Base):
    """Two-dimensional numeric array with linear algebra operations."""

    def __init__(self, data: np.ndarray) -> None:
        super().__init__(data)

        if self.ndim != 2:
            raise ValueError("Matrix must be 2D")

    def __str__(self) -> str:
        return format_array(self.data)

    def __repr__(self) -> str:
        return f"Matrix({format_array(self.data)})"

    def __bool__(self) -> bool:
        return self.size > 0

    def __float__(self):
        if self.size != 1:
            raise ValueError(f"Cannot convert Matrix with shape {self.shape} to float")
        return float(self.data.reshape(-1)[0])

    def __complex__(self):
        if self.size != 1:
            raise ValueError(f"Cannot convert Matrix with shape {self.shape} to complex")
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
        return f"Matrix({formatted})"

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
        if result.ndim == 1:
            return Vector(result)
        return Matrix(result)

    def __eq__(self, other):
        if isinstance(other, Matrix):
            return Matrix(isclose(self.data, other.data))
        return NotImplemented

    def __ne__(self, other):
        if isinstance(other, Matrix):
            return Matrix(~isclose(self.data, other.data))
        return NotImplemented

    def __lt__(self, other):
        if isinstance(other, Matrix):
            return Matrix(self.data < other.data)
        return NotImplemented

    def __le__(self, other):
        if isinstance(other, Matrix):
            return Matrix(self.data <= other.data)
        return NotImplemented

    def __gt__(self, other):
        if isinstance(other, Matrix):
            return Matrix(self.data > other.data)
        return NotImplemented

    def __ge__(self, other):
        if isinstance(other, Matrix):
            return Matrix(self.data >= other.data)
        return NotImplemented

    def __neg__(self):
        return Matrix(-self.data)
    
    def __pow__(self, exponent):
        """ repeated matrix multiplication for integer exponents, elementwise for others"""
        
        if isinstance(exponent, int):
            if exponent < 0:
                return self.inverse() ** (-exponent)
            result = Matrix(np.eye(self.shape[0]))
            for _ in range(exponent):
                result = result @ self
            return result
        if isinstance(exponent, (float, np.floating)):
            return Matrix(self.data ** exponent)
        if isinstance(exponent, Matrix):
            if self.shape != exponent.shape:
                raise ValueError("Matrix shapes must be the same for exponentiation")
            return Matrix(self.data ** exponent.data)
        return NotImplemented

    def __pos__(self):
        return Matrix(+self.data)

    def __abs__(self):
        return Matrix(np.abs(self.data))

    def __copy__(self) -> Matrix:
        return Matrix(self.data.copy())

    def __deepcopy__(self, memo) -> Matrix:
        return Matrix(copy.deepcopy(self.data, memo))

    def copy(self) -> Matrix:
        return self.__copy__()

    def __add__(self, other):
        if not isinstance(other, Matrix):
            return NotImplemented
        if self.shape != other.shape:
            raise ValueError("Matrix shapes must be the same for addition")
        rows, cols = self.shape
        result = np.empty((rows, cols), dtype=np.float64)
        for i in range(rows):
            for j in range(cols):
                result[i, j] = self.data[i, j] + other.data[i, j]
        return Matrix(result)

    def __sub__(self, other):
        if not isinstance(other, Matrix):
            return NotImplemented
        if self.shape != other.shape:
            raise ValueError("Matrix shapes must be the same for subtraction")
        rows, cols = self.shape
        result = np.empty((rows, cols), dtype=np.float64)
        for i in range(rows):
            for j in range(cols):
                result[i, j] = self.data[i, j] - other.data[i, j]
        return Matrix(result)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            rows, cols = self.shape
            result = np.empty((rows, cols), dtype=np.float64)
            for i in range(rows):
                for j in range(cols):
                    result[i, j] = self.data[i, j] * other
            return Matrix(result)
        if isinstance(other, Matrix):
            if self.shape != other.shape:
                raise ValueError("Matrix shapes must be the same for multiplication")
            rows, cols = self.shape
            result = np.empty((rows, cols), dtype=np.float64)
            for i in range(rows):
                for j in range(cols):
                    result[i, j] = self.data[i, j] * other.data[i, j]
            return Matrix(result)
        return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            if isclose(other, 0.0):
                raise ZeroDivisionError("Cannot divide by zero")
            rows, cols = self.shape
            result = np.empty((rows, cols), dtype=np.float64)
            for i in range(rows):
                for j in range(cols):
                    result[i, j] = clean(self.data[i, j] / other)
            return Matrix(result)
        if isinstance(other, Matrix):
            if self.shape != other.shape:
                raise ValueError("Matrix shapes must be the same for elementwise division")
            if np.any(isclose(other.data, 0.0)):
                raise ZeroDivisionError("Cannot divide by zero")
            rows, cols = self.shape
            result = np.empty((rows, cols), dtype=np.float64)
            for i in range(rows):
                for j in range(cols):
                    result[i, j] = clean(self.data[i, j] / other.data[i, j])
            return Matrix(result)
        return NotImplemented

    def __matmul__(self, other):
        if isinstance(other, Matrix):
            if self.shape[1] != other.shape[0]:
                raise ValueError("Matrix shapes are not aligned for matrix multiplication")
            rows, inner = self.shape
            _, cols = other.shape
            result = np.zeros((rows, cols), dtype=np.float64)
            for i in range(rows):
                for j in range(cols):
                    total = 0.0
                    for k in range(inner):
                        total += self.data[i, k] * other.data[k, j]
                    result[i, j] = total
            return Matrix(result)
        if isinstance(other, Vector):
            if self.shape[1] != other.shape[0]:
                raise ValueError("Matrix shapes are not aligned for matrix multiplication")
            rows, cols = self.shape
            result = np.zeros(rows, dtype=np.float64)
            for i in range(rows):
                total = 0.0
                for j in range(cols):
                    total += self.data[i, j] * other.data[j]
                result[i] = total
            return Vector(result)
        return NotImplemented

    def __radd__(self, other):
        return self.__add__(other)

    def __rsub__(self, other):
        if not isinstance(other, Matrix):
            return NotImplemented
        if self.shape != other.shape:
            raise ValueError("Matrix shapes must be the same for subtraction")
        rows, cols = self.shape
        result = np.empty((rows, cols), dtype=np.float64)
        for i in range(rows):
            for j in range(cols):
                result[i, j] = other.data[i, j] - self.data[i, j]
        return Matrix(result)

    def __rmul__(self, other):
        if isinstance(other, (int, float)):
            rows, cols = self.shape
            result = np.empty((rows, cols), dtype=np.float64)
            for i in range(rows):
                for j in range(cols):
                    result[i, j] = other * self.data[i, j]
            return Matrix(result)
        if isinstance(other, Matrix):
            if self.shape != other.shape:
                raise ValueError("Matrix shapes must be the same for multiplication")
            rows, cols = self.shape
            result = np.empty((rows, cols), dtype=np.float64)
            for i in range(rows):
                for j in range(cols):
                    result[i, j] = other.data[i, j] * self.data[i, j]
            return Matrix(result)
        return NotImplemented

    def __rtruediv__(self, other):
        if isinstance(other, (int, float)):
            if np.any(isclose(self.data, 0.0)):
                raise ZeroDivisionError("Cannot divide by zero")
            rows, cols = self.shape
            result = np.empty((rows, cols), dtype=np.float64)
            for i in range(rows):
                for j in range(cols):
                    result[i, j] = clean(other / self.data[i, j])
            return Matrix(result)
        if isinstance(other, Matrix):
            if self.shape != other.shape:
                raise ValueError("Matrix shapes must be the same for elementwise division")
            if np.any(isclose(self.data, 0.0)):
                raise ZeroDivisionError("Cannot divide by zero")
            rows, cols = self.shape
            result = np.empty((rows, cols), dtype=np.float64)
            for i in range(rows):
                for j in range(cols):
                    result[i, j] = clean(other.data[i, j] / self.data[i, j])
            return Matrix(result)
        return NotImplemented

    def __rmatmul__(self, other):
        if not isinstance(other, Matrix):
            return NotImplemented
        if other.shape[1] != self.shape[0]:
            raise ValueError("Matrix shapes are not aligned for matrix multiplication")
        rows, inner = other.shape
        _, cols = self.shape
        result = np.zeros((rows, cols), dtype=np.float64)
        for i in range(rows):
            for j in range(cols):
                total = 0.0
                for k in range(inner):
                    total += other.data[i, k] * self.data[k, j]
                result[i, j] = total
        return Matrix(result)

    def __iadd__(self, other):
        if not isinstance(other, Matrix):
            return NotImplemented
        if self.shape != other.shape:
            raise ValueError("Matrix shapes must be the same for addition")
        rows, cols = self.shape
        for i in range(rows):
            for j in range(cols):
                self.data[i, j] += other.data[i, j]
        return self

    def __isub__(self, other):
        if not isinstance(other, Matrix):
            return NotImplemented
        if self.shape != other.shape:
            raise ValueError("Matrix shapes must be the same for subtraction")
        rows, cols = self.shape
        for i in range(rows):
            for j in range(cols):
                self.data[i, j] -= other.data[i, j]
        return self

    def __imul__(self, other):
        if isinstance(other, (int, float)):
            rows, cols = self.shape
            for i in range(rows):
                for j in range(cols):
                    self.data[i, j] *= other
            return self
        if isinstance(other, Matrix):
            if self.shape != other.shape:
                raise ValueError("Matrix shapes must be the same for multiplication")
            rows, cols = self.shape
            for i in range(rows):
                for j in range(cols):
                    self.data[i, j] *= other.data[i, j]
            return self
        return NotImplemented

    def __itruediv__(self, other):
        if isinstance(other, (int, float)):
            if np.isclose(other, 0.0, rtol=r_tol, atol=a_tol):
                raise ZeroDivisionError("Cannot divide by zero")
            rows, cols = self.shape
            for i in range(rows):
                for j in range(cols):
                    self.data[i, j] /= other
            return self
        if isinstance(other, Matrix):
            if self.shape != other.shape:
                raise ValueError("Matrix shapes must be the same for elementwise division")
            if np.any(isclose(other.data, 0.0)):
                raise ZeroDivisionError("Cannot divide by zero")
            rows, cols = self.shape
            for i in range(rows):
                for j in range(cols):
                    self.data[i, j] = clean(self.data[i, j] / other.data[i, j])
            return self
        return NotImplemented

    def __imatmul__(self, other):
        if not isinstance(other, Matrix):
            return NotImplemented
        if self.shape[1] != other.shape[0]:
            raise ValueError("Matrix shapes are not aligned for matrix multiplication")
        rows, inner = self.shape
        _, cols = other.shape
        result = np.zeros((rows, cols), dtype=np.float64)
        for i in range(rows):
            for j in range(cols):
                total = 0.0
                for k in range(inner):
                    total += self.data[i, k] * other.data[k, j]
                result[i, j] = total
        self.data = result
        return self

    def transpose(self) -> Matrix:
        rows, cols = self.shape
        result = np.empty((cols, rows), dtype=np.float64)
        for i in range(rows):
            for j in range(cols):
                result[j, i] = self.data[i, j]
        return Matrix(result)

    @property
    def T(self) -> Matrix:
        return self.transpose()

    def determinant(self) -> float:
        if self.shape[0] != self.shape[1]:
            raise ValueError("Determinant is only defined for square matrices")
        return float(np.linalg.det(self.data))

    def inverse(self) -> Matrix:
        if self.shape[0] != self.shape[1]:
            raise ValueError("Inverse is only defined for square matrices")
        return Matrix(np.linalg.inv(self.data))

    def trace(self) -> float:
        if self.shape[0] != self.shape[1]:
            raise ValueError("Trace is only defined for square matrices")
        total = 0.0
        for i in range(self.shape[0]):
            total += self.data[i, i]
        return float(total)

    def eigenvalues(self) -> np.ndarray:
        if self.shape[0] != self.shape[1]:
            raise ValueError("Eigenvalues are only defined for square matrices")
        return np.linalg.eigvals(self.data)

    def eigenvectors(self) -> tuple[Vector, Matrix]:
        if self.shape[0] != self.shape[1]:
            raise ValueError("Eigenvectors are only defined for square matrices")
        values, vectors = np.linalg.eig(self.data)
        return Vector(values), Matrix(vectors)

    def solve(self, b: Vector) -> Vector:
        if self.shape[0] != self.shape[1]:
            raise ValueError("Solving is only defined for square matrices")
        if not isinstance(b, Vector):
            raise TypeError("b must be a Vector")
        if self.shape[0] != b.size:
            raise ValueError("Matrix shapes are not aligned for matrix multiplication")
        return Vector(np.linalg.solve(self.data, b.data))
