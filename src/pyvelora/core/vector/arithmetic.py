from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyvelora.core.vector.vector import Vector
    from pyvelora.core.matrix import Matrix

# Arithmetic operations for Vector class
class VectorArithmetic:
    """Arithmetic operations mixin for Vector class."""

    def __add__(self, other: Vector) -> Vector:
        """Add two vectors."""
        if not isinstance(other, type(self)):
            raise TypeError("Both operands must be of type Vector.")
        if self.size != other.size:
            raise ValueError("Vectors must be of the same size for addition.")
        return type(self)([a + b for a, b in zip(self.data, other.data)])

    def __sub__(self, other: Vector) -> Vector:
        """Subtract vector b from vector a."""
        if not isinstance(other, type(self)):
            raise TypeError("Both operands must be of type Vector.")
        if self.size != other.size:
            raise ValueError("Vectors must be of the same size for subtraction.")
        return type(self)([a - b for a, b in zip(self.data, other.data)])

    def __mul__(self, other: Vector | float | int) -> Vector:
        """Element-wise multiplication of two vectors or scalar multiplication."""
        if not isinstance(other, (type(self), float, int)):
            raise TypeError("Operands must be of type Vector or a scalar (float or int).")
        if isinstance(other, type(self)) and self.size != other.size:
            raise ValueError("Vectors must be of the same size for multiplication.")
        if isinstance(other, (float, int)):
            return type(self)([a * other for a in self.data])
        return type(self)([a * b for a, b in zip(self.data, other.data)])

    def __truediv__(self, other: Vector | float | int) -> Vector:
        """Element-wise division of two vectors or scalar division."""
        if not isinstance(other, (type(self), float, int)):
            raise TypeError("Operands must be of type Vector or a scalar (float or int).")
        if isinstance(other, type(self)) and self.size != other.size:
            raise ValueError("Vectors must be of the same size for division.")
        if isinstance(other, (float, int)):
            if other == 0:
                raise ValueError("Cannot divide by zero.")
            return type(self)([a / other for a in self.data])
        if any(b == 0 for b in other.data):
            raise ValueError("Cannot divide by zero in vector division.")
        return type(self)([a / b for a, b in zip(self.data, other.data)])

    def __matmul__(self, other: Vector | Matrix) -> float | Vector:
        """Matrix multiplication of two vectors (dot product) or vector-matrix multiplication."""
        from pyvelora.core.matrix import Matrix
        if not isinstance(other, (type(self), Matrix)):
            raise TypeError("Operands must be of type Vector or Matrix.")
        if isinstance(other, type(self)):
            if self.size != other.size:
                raise ValueError("Vectors must be of the same size for dot product.")
            return float(sum(a * b for a, b in zip(self.data, other.data)))
        if self.size != other.shape[0]:
            raise ValueError("Vector size must match the number of rows in the matrix for multiplication.")
        return type(self)([sum(a * b for a, b in zip(self.data, col)) for col in zip(*other.data)])

# right-hand vector operations

    def __radd__(self, other: Vector) -> Vector:
        """Right-hand addition of two vectors."""
        if not isinstance(other, type(self)):
            raise TypeError("Both operands must be of type Vector.")
        if self.size != other.size:
            raise ValueError("Vectors must be of the same size for addition.")
        return type(self)([a + b for a, b in zip(other.data, self.data)])

    def __rsub__(self, other: Vector) -> Vector:
        """Right-hand subtraction of two vectors."""
        if not isinstance(other, type(self)):
            raise TypeError("Both operands must be of type Vector.")
        if self.size != other.size:
            raise ValueError("Vectors must be of the same size for subtraction.")
        return type(self)([a - b for a, b in zip(other.data, self.data)])

    def __rmul__(self, other: Vector | float | int) -> Vector:
        """Right-hand multiplication of a vector by another vector or scalar."""
        if isinstance(other, (float, int)):
            return type(self)([other * a for a in self.data])
        return type(self)([a * b for a, b in zip(other.data, self.data)])

    def __rtruediv__(self, other: Vector | float | int) -> Vector:
        """Right-hand division of a vector by another vector or scalar."""
        if isinstance(other, (float, int)):
            if any(a == 0 for a in self.data):
                raise ValueError("Cannot divide by zero in vector division.")
            return type(self)([other / a for a in self.data])
        if self.size != other.size:
            raise ValueError("Vectors must be of the same size for division.")
        if any(a == 0 for a in self.data):
            raise ValueError("Cannot divide by zero in vector division.")
        return type(self)([a / b for a, b in zip(other.data, self.data)])

    def __rmatmul__(self, other: Vector | Matrix) -> float | Vector:
        """Right-hand matrix multiplication of a vector by another vector or matrix."""
        if isinstance(other, type(self)):
            if other.size != self.size:
                raise ValueError("Vectors must be of the same size for dot product.")
            return float(sum(a * b for a, b in zip(other.data, self.data)))
        if other.shape[1] != self.size:
            raise ValueError("Number of columns in the matrix must match the size of the vector for multiplication.")
        return type(self)([sum(a * b for a, b in zip(row, self.data)) for row in other.data])

# in-place vector operations

    def __iadd__(self, other: Vector) -> Vector:
        """In-place addition of two vectors."""
        if not isinstance(other, type(self)):
            raise TypeError("Both operands must be of type Vector.")
        if self.size != other.size:
            raise ValueError("Vectors must be of the same size for in-place addition.")
        self.data = [a + b for a, b in zip(self.data, other.data)]
        return self

    def __isub__(self, other: Vector) -> Vector:
        """In-place subtraction of two vectors."""
        if not isinstance(other, type(self)):
            raise TypeError("Both operands must be of type Vector.")
        if self.size != other.size:
            raise ValueError("Vectors must be of the same size for in-place subtraction.")
        self.data = [a - b for a, b in zip(self.data, other.data)]
        return self

    def __imul__(self, other: Vector | float | int) -> Vector:
        """In-place multiplication of a vector by another vector or scalar."""
        if not isinstance(other, (type(self), float, int)):
            raise TypeError("Operands must be of type Vector or a scalar (float or int).")
        if isinstance(other, type(self)) and self.size != other.size:
            raise ValueError("Vectors must be of the same size for in-place multiplication.")
        if isinstance(other, (float, int)):
            self.data = [a * other for a in self.data]
        else:
            self.data = [a * b for a, b in zip(self.data, other.data)]
        return self

    def __itruediv__(self, other: Vector | float | int) -> Vector:
        """In-place division of a vector by another vector or scalar."""
        if not isinstance(other, (type(self), float, int)):
            raise TypeError("Operands must be of type Vector or a scalar (float or int).")
        if isinstance(other, type(self)) and self.size != other.size:
            raise ValueError("Vectors must be of the same size for in-place division.")
        if isinstance(other, (float, int)):
            if other == 0:
                raise ValueError("Cannot divide by zero.")
            self.data = [a / other for a in self.data]
        else:
            if any(a == 0 for a in other.data):
                raise ValueError("Cannot divide by zero in vector division.")
            self.data = [a / b for a, b in zip(self.data, other.data)]
        return self

    def __imatmul__(self, other: Vector | Matrix) -> Vector:
        """In-place matrix multiplication of a vector by another vector or matrix."""
        from pyvelora.core.matrix import Matrix
        if not isinstance(other, (type(self), Matrix)):
            raise TypeError("Operands must be of type Vector or Matrix.")
        if isinstance(other, type(self)):
            if self.size != other.size:
                raise ValueError("Vectors must be of the same size for in-place dot product.")
            self.data = sum(a * b for a, b in zip(self.data, other.data))
        else:
            if self.size != other.shape[0]:
                raise ValueError("Vector size must match the number of rows in the matrix for in-place multiplication.")
            self.data = [sum(a * b for a, b in zip(self.data, col)) for col in zip(*other.data)]
        return self

# unary vector operations (e.g., negation)

    def __neg__(self) -> Vector:
        """Negation of a vector."""
        return type(self)([-a for a in self.data])

    def __pos__(self) -> Vector:
        """Unary plus (identity) of a vector."""
        return type(self)([+a for a in self.data])

    def __abs__(self) -> Vector:
        """Absolute value of a vector."""
        return type(self)([abs(a) for a in self.data])


