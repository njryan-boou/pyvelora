from __future__ import annotations
from pyvelora.core import Matrix, Vector
from pyvelora.linalg.constructors import identity
from pyvelora.linalg.products import matmul
from pyvelora.linalg.rref import rref


def trace(A: Matrix) -> float:
    """Compute the trace of a square matrix."""
    if not isinstance(A, Matrix) or A.shape[0] != A.shape[1]:
        raise ValueError("Trace is only defined for square matrices")
    return float(sum(A[i, i] for i in range(A.shape[0])))

def rank(A: Matrix) -> int:
    """Compute the rank of a matrix."""
    if not isinstance(A, Matrix):
        raise TypeError("Argument must be a Matrix")
    _, pivot_cols = rref(A)
    return len(pivot_cols)

def minor(A: Matrix, i: int, j: int) -> float:
    """Compute the minor of a matrix A at position (i, j)."""
    if not isinstance(A, Matrix):
        raise TypeError("Argument must be a Matrix")
    if i < 0 or i >= A.shape[0] or j < 0 or j >= A.shape[1]:
        raise IndexError("Row and column indices must be within matrix dimensions")
    submatrix = [
        [value for col_index, value in enumerate(row) if col_index != j]
        for row_index, row in enumerate(A.data)
        if row_index != i
    ]
    return float(determinant(Matrix(submatrix)))

def cofactor(A: Matrix, i: int, j: int) -> float:
    """Compute the cofactor of a matrix A at position (i, j)."""
    if not isinstance(A, Matrix):
        raise TypeError("Argument must be a Matrix")
    return ((-1) ** (i + j)) * minor(A, i, j)

def cofactor_matrix(A: Matrix) -> Matrix:
    """Compute the cofactor matrix of a square matrix A."""
    if not isinstance(A, Matrix) or A.shape[0] != A.shape[1]:
        raise ValueError("Cofactor matrix is only defined for square matrices")
    n = A.shape[0]
    cofactor_data = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            cofactor_data[i][j] = cofactor(A, i, j)
    return Matrix(cofactor_data)

def adjugate(A: Matrix) -> Matrix:
    """Compute the adjugate of a square matrix A."""
    if not isinstance(A, Matrix) or A.shape[0] != A.shape[1]:
        raise ValueError("Adjugate is only defined for square matrices")
    cofactor_data = cofactor_matrix(A).data
    return Matrix([[cofactor_data[row][col] for row in range(A.shape[0])] for col in range(A.shape[1])])

def determinant(A: Matrix) -> float:
    """Compute the determinant of a square matrix A."""
    if not isinstance(A, Matrix) or A.shape[0] != A.shape[1]:
        raise ValueError("Determinant is only defined for square matrices")
    n = A.shape[0]
    rows = [[float(value) for value in row] for row in A.data]
    sign = 1.0

    for col in range(n):
        pivot_row = max(range(col, n), key=lambda row: abs(rows[row][col]))
        if abs(rows[pivot_row][col]) < 1e-12:
            return 0.0
        if pivot_row != col:
            rows[col], rows[pivot_row] = rows[pivot_row], rows[col]
            sign *= -1.0
        pivot = rows[col][col]
        for row in range(col + 1, n):
            factor = rows[row][col] / pivot
            for inner_col in range(col, n):
                rows[row][inner_col] -= factor * rows[col][inner_col]

    result = sign
    for index in range(n):
        result *= rows[index][index]
    return float(result)

def inverse(A: Matrix) -> Matrix:
    """Compute the inverse of a square, invertible matrix A."""
    if not isinstance(A, Matrix) or A.shape[0] != A.shape[1]:
        raise ValueError("Inverse is only defined for square matrices")
    n = A.shape[0]
    augmented = [
        [float(value) for value in A.data[row]] + [float(value) for value in identity(n).data[row]]
        for row in range(n)
    ]

    for col in range(n):
        pivot_row = max(range(col, n), key=lambda row: abs(augmented[row][col]))
        if abs(augmented[pivot_row][col]) < 1e-12:
            raise ValueError("Matrix is singular and cannot be inverted")
        augmented[col], augmented[pivot_row] = augmented[pivot_row], augmented[col]

        pivot = augmented[col][col]
        augmented[col] = [value / pivot for value in augmented[col]]

        for row in range(n):
            if row == col:
                continue
            factor = augmented[row][col]
            if abs(factor) < 1e-12:
                continue
            augmented[row] = [value - factor * pivot_value for value, pivot_value in zip(augmented[row], augmented[col])]

    return Matrix([row[n:] for row in augmented])
