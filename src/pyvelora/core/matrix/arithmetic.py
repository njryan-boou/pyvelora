from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyvelora.core.matrix.matrix import Matrix
    from pyvelora.core.vector.vector import Vector


class MatrixArithmetic:
    """Arithmetic operations mixin for Matrix class."""

    def __neg__(self) -> Matrix:
        return type(self)([[-value for value in row] for row in self.data])

    def __pos__(self) -> Matrix:
        return type(self)([[+value for value in row] for row in self.data])

    def __abs__(self) -> Matrix:
        return type(self)([[abs(value) for value in row] for row in self.data])

    def __pow__(self, exponent) -> Matrix:
        if isinstance(exponent, int):
            if self.shape[0] != self.shape[1]:
                raise ValueError("Matrix exponentiation is only defined for square matrices")
            size = self.shape[0]
            result = [[1.0 if row == col else 0.0 for col in range(size)] for row in range(size)]
            base = [row[:] for row in self.data]
            power = exponent
            while power > 0:
                if power % 2 == 1:
                    result = [
                        [sum(result[row][inner] * base[inner][col] for inner in range(len(base))) for col in range(len(base[0]))]
                        for row in range(len(result))
                    ]
                base = [
                    [sum(base[row][inner] * base[inner][col] for inner in range(len(base))) for col in range(len(base[0]))]
                    for row in range(len(base))
                ]
                power //= 2
            return type(self)(result)
        if isinstance(exponent, type(self)):
            if self.shape != exponent.shape:
                raise ValueError("Matrix shapes must be the same for elementwise exponentiation")
            return type(self)([[left ** right for left, right in zip(left_row, right_row)] for left_row, right_row in zip(self.data, exponent.data)])
        return NotImplemented

    def __add__(self, other) -> Matrix:
        if not isinstance(other, type(self)):
            return NotImplemented
        if self.shape != other.shape:
            raise ValueError("Matrix shapes must be the same for addition")
        return type(self)([[left + right for left, right in zip(left_row, right_row)] for left_row, right_row in zip(self.data, other.data)])

    def __sub__(self, other) -> Matrix:
        if not isinstance(other, type(self)):
            return NotImplemented
        if self.shape != other.shape:
            raise ValueError("Matrix shapes must be the same for subtraction")
        return type(self)([[left - right for left, right in zip(left_row, right_row)] for left_row, right_row in zip(self.data, other.data)])

    def __mul__(self, other) -> Matrix:
        if isinstance(other, (int, float)):
            return type(self)([[value * other for value in row] for row in self.data])
        if isinstance(other, type(self)):
            if self.shape != other.shape:
                raise ValueError("Matrix shapes must be the same for multiplication")
            return type(self)([[left * right for left, right in zip(left_row, right_row)] for left_row, right_row in zip(self.data, other.data)])
        return NotImplemented

    def __truediv__(self, other) -> Matrix:
        if isinstance(other, (int, float)):
            if other == 0:
                raise ZeroDivisionError("Cannot divide by zero")
            return type(self)([[value / other for value in row] for row in self.data])
        if isinstance(other, type(self)):
            if self.shape != other.shape:
                raise ValueError("Matrix shapes must be the same for elementwise division")
            if any(any(value == 0 for value in row) for row in other.data):
                raise ZeroDivisionError("Cannot divide by zero")
            return type(self)([[left / right for left, right in zip(left_row, right_row)] for left_row, right_row in zip(self.data, other.data)])
        return NotImplemented

    def __matmul__(self, other) -> Matrix | Vector:
        if isinstance(other, type(self)):
            if self.shape[1] != other.shape[0]:
                raise ValueError("Matrix shapes are not aligned for matrix multiplication")
            return type(self)([
                [sum(self.data[row][inner] * other.data[inner][col] for inner in range(len(other.data))) for col in range(len(other.data[0]))]
                for row in range(len(self.data))
            ])
        from pyvelora.core.vector.vector import Vector
        if isinstance(other, Vector):
            if self.shape[1] != other.shape[0]:
                raise ValueError("Matrix shapes are not aligned for matrix multiplication")
            return Vector([sum(left * right for left, right in zip(row, other.data)) for row in self.data])
        return NotImplemented

    def __radd__(self, other) -> Matrix:
        if not isinstance(other, type(self)):
            return NotImplemented
        if self.shape != other.shape:
            raise ValueError("Matrix shapes must be the same for addition")
        return type(self)([[left + right for left, right in zip(left_row, right_row)] for left_row, right_row in zip(other.data, self.data)])

    def __rsub__(self, other) -> Matrix:
        if not isinstance(other, type(self)):
            return NotImplemented
        if self.shape != other.shape:
            raise ValueError("Matrix shapes must be the same for subtraction")
        return type(self)([[left - right for left, right in zip(left_row, right_row)] for left_row, right_row in zip(other.data, self.data)])

    def __rmul__(self, other) -> Matrix:
        if isinstance(other, (int, float)):
            return type(self)([[other * value for value in row] for row in self.data])
        if isinstance(other, type(self)):
            if self.shape != other.shape:
                raise ValueError("Matrix shapes must be the same for multiplication")
            return type(self)([[left * right for left, right in zip(left_row, right_row)] for left_row, right_row in zip(other.data, self.data)])
        return NotImplemented

    def __rtruediv__(self, other) -> Matrix:
        if isinstance(other, (int, float)):
            if any(any(value == 0 for value in row) for row in self.data):
                raise ZeroDivisionError("Cannot divide by zero")
            return type(self)([[other / value for value in row] for row in self.data])
        if isinstance(other, type(self)):
            if self.shape != other.shape:
                raise ValueError("Matrix shapes must be the same for elementwise division")
            if any(any(value == 0 for value in row) for row in self.data):
                raise ZeroDivisionError("Cannot divide by zero")
            return type(self)([[left / right for left, right in zip(left_row, right_row)] for left_row, right_row in zip(other.data, self.data)])
        return NotImplemented

    def __rmatmul__(self, other) -> Matrix:
        if not isinstance(other, type(self)):
            return NotImplemented
        if other.shape[1] != self.shape[0]:
            raise ValueError("Matrix shapes are not aligned for matrix multiplication")
        return type(self)([
            [sum(other.data[row][inner] * self.data[inner][col] for inner in range(len(self.data))) for col in range(len(self.data[0]))]
            for row in range(len(other.data))
        ])

    def __iadd__(self, other) -> Matrix:
        if not isinstance(other, type(self)):
            return NotImplemented
        if self.shape != other.shape:
            raise ValueError("Matrix shapes must be the same for addition")
        self.data = [[left + right for left, right in zip(left_row, right_row)] for left_row, right_row in zip(self.data, other.data)]
        return self

    def __isub__(self, other) -> Matrix:
        if not isinstance(other, type(self)):
            return NotImplemented
        if self.shape != other.shape:
            raise ValueError("Matrix shapes must be the same for subtraction")
        self.data = [[left - right for left, right in zip(left_row, right_row)] for left_row, right_row in zip(self.data, other.data)]
        return self

    def __imul__(self, other) -> Matrix:
        if isinstance(other, (int, float)):
            self.data = [[value * other for value in row] for row in self.data]
            return self
        if isinstance(other, type(self)):
            if self.shape != other.shape:
                raise ValueError("Matrix shapes must be the same for multiplication")
            self.data = [[left * right for left, right in zip(left_row, right_row)] for left_row, right_row in zip(self.data, other.data)]
            return self
        return NotImplemented

    def __itruediv__(self, other) -> Matrix:
        if isinstance(other, (int, float)):
            if other == 0.0:
                raise ZeroDivisionError("Cannot divide by zero")
            self.data = [[value / other for value in row] for row in self.data]
            return self
        if isinstance(other, type(self)):
            if self.shape != other.shape:
                raise ValueError("Matrix shapes must be the same for elementwise division")
            if any(any(value == 0 for value in row) for row in other.data):
                raise ZeroDivisionError("Cannot divide by zero")
            self.data = [[left / right for left, right in zip(left_row, right_row)] for left_row, right_row in zip(self.data, other.data)]
            return self
        return NotImplemented

    def __imatmul__(self, other) -> Matrix:
        if not isinstance(other, type(self)):
            return NotImplemented
        if self.shape[1] != other.shape[0]:
            raise ValueError("Matrix shapes are not aligned for matrix multiplication")
        self.data = [
            [sum(self.data[row][inner] * other.data[inner][col] for inner in range(len(other.data))) for col in range(len(other.data[0]))]
            for row in range(len(self.data))
        ]
        return self
