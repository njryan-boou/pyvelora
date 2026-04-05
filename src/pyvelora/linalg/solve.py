from __future__ import annotations
from pyvelora.core import Vector, Matrix
from pyvelora.linalg.checks import is_lower_triangular, is_upper_triangular
from pyvelora.linalg.decompositions import lu_decomposition

def forward_substitution(L: Matrix, b: Vector) -> Vector:
    """Solve Lx = b for x where L is a lower triangular matrix."""
    if not isinstance(L, Matrix) or not isinstance(b, Vector):
        raise TypeError("First argument must be a Matrix and second argument must be a Vector")
    if L.shape[0] != L.shape[1]:
        raise ValueError("Matrix must be square for forward substitution")
    if L.shape[0] != b.shape[0]:
        raise ValueError("Incompatible dimensions for forward substitution")
    values = [0.0] * L.shape[0]
    for row in range(L.shape[0]):
        total = b.data[row]
        for col in range(row):
            total -= L.data[row][col] * values[col]
        pivot = L.data[row][row]
        if pivot == 0:
            raise ValueError("Matrix is singular for forward substitution")
        values[row] = total / pivot
    return Vector(values)

def backward_substitution(U: Matrix, b: Vector) -> Vector:
    """Solve Ux = b for x where U is an upper triangular matrix."""
    if not isinstance(U, Matrix) or not isinstance(b, Vector):
        raise TypeError("First argument must be a Matrix and second argument must be a Vector")
    if U.shape[0] != U.shape[1]:
        raise ValueError("Matrix must be square for backward substitution")
    if U.shape[0] != b.shape[0]:
        raise ValueError("Incompatible dimensions for backward substitution")
    values = [0.0] * U.shape[0]
    for row in range(U.shape[0] - 1, -1, -1):
        total = b.data[row]
        for col in range(row + 1, U.shape[1]):
            total -= U.data[row][col] * values[col]
        pivot = U.data[row][row]
        if pivot == 0:
            raise ValueError("Matrix is singular for backward substitution")
        values[row] = total / pivot
    return Vector(values)

def solve_lu(A: Matrix, b: Vector) -> Vector:
    """Solve Ax = b using LU decomposition."""
    if not isinstance(A, Matrix) or not isinstance(b, Vector):
        raise TypeError("First argument must be a Matrix and second argument must be a Vector")
    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix must be square for LU decomposition")
    if A.shape[0] != b.shape[0]:
        raise ValueError("Incompatible dimensions for LU decomposition")
    
    L, U = lu_decomposition(A)
    y = forward_substitution(L, b)
    return backward_substitution(U, y)

def solve_linear_system(A: Matrix, b: Vector) -> Vector:
    """Solve Ax = b for x using an appropriate method based on the properties of A."""
    if not isinstance(A, Matrix) or not isinstance(b, Vector):
        raise TypeError("First argument must be a Matrix and second argument must be a Vector")
    if A.shape[0] != b.shape[0]:
        raise ValueError("Incompatible dimensions for solving linear system")
    
    if is_upper_triangular(A):
        return backward_substitution(A, b)
    elif is_lower_triangular(A):
        return forward_substitution(A, b)
    else:
        return solve_lu(A, b)