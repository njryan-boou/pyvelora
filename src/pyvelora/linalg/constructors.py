from __future__ import annotations

from pyvelora.core import Vector, Matrix

def zeros(rows: int, cols: int) -> Matrix:
    """Create a matrix of zeros with the specified shape."""
    if not isinstance(rows, int) or not isinstance(cols, int):
        raise TypeError("Rows and columns must be integers")
    if rows <= 0 or cols <= 0:
        raise ValueError("Rows and columns must be positive integers")
    return Matrix([[0 for _ in range(cols)] for _ in range(rows)])

def ones(rows: int, cols: int) -> Matrix:
    """Create a matrix of ones with the specified shape."""
    if not isinstance(rows, int) or not isinstance(cols, int):
        raise TypeError("Rows and columns must be integers")
    if rows <= 0 or cols <= 0:
        raise ValueError("Rows and columns must be positive integers")
    return Matrix([[1 for _ in range(cols)] for _ in range(rows)])

def full(rows: int, cols: int, fill_value: float) -> Matrix:
    """Create a matrix filled with the specified value."""
    if not isinstance(rows, int) or not isinstance(cols, int):
        raise TypeError("Rows and columns must be integers")
    if rows <= 0 or cols <= 0:
        raise ValueError("Rows and columns must be positive integers")
    return Matrix([[fill_value for _ in range(cols)] for _ in range(rows)])

def identity(n: int) -> Matrix:
    """Create an identity matrix of the specified size."""
    if not isinstance(n, int):
        raise TypeError("Size of identity matrix must be an integer")
    if n <= 0:
        raise ValueError("Size of identity matrix must be a positive integer")
    return Matrix([[1 if i == j else 0 for j in range(n)] for i in range(n)])

def diagonal(values: list | Vector) -> Matrix:
    """Create a diagonal matrix from the given vector or list of values."""
    if isinstance(values, Vector):
        values = values.data
    elif not isinstance(values, list):
        raise TypeError("Values must be a list or a Vector")
    size = len(values)
    return Matrix([[values[i] if i == j else 0 for j in range(size)] for i in range(size)])

def from_rows(rows: list[Vector]) -> Matrix:
    """Create a matrix from a list of row vectors."""
    return Matrix([row.data[:] for row in rows])

def from_cols(cols: list[Vector]) -> Matrix:
    """Create a matrix from a list of column vectors."""
    return Matrix([[col.data[row] for col in cols] for row in range(len(cols[0].data))])

