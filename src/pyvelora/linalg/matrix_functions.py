from __future__ import annotations
from pyvelora.core import Vector, Matrix
from pyvelora.linalg.basic import add, scalar_multiply
from pyvelora.linalg.constructors import identity
from pyvelora.linalg.properties import inverse
from pyvelora.linalg.products import matmul

def matrix_exponential(A: Matrix) -> Matrix:
    """Compute the matrix exponential of A."""
    if not isinstance(A, Matrix):
        raise TypeError("Argument must be a Matrix")
    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix exponential is only defined for square matrices")
    result = identity(A.shape[0])
    term = identity(A.shape[0])
    for iteration in range(1, 30):
        term = scalar_multiply(matmul(term, A), 1 / iteration)
        result = add(result, term)
        if max(abs(value) for row in term.data for value in row) < 1e-12:
            break
    return result

def matrix_power(A: Matrix, n: int) -> Matrix:
    """Compute the nth power of a square matrix A."""
    if not isinstance(A, Matrix):
        raise TypeError("First argument must be a Matrix")
    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix power is only defined for square matrices")
    if not isinstance(n, int):
        raise ValueError("Second argument must be an integer")

    if n < 0:
        # A^(-n) = (A^-1)^n
        return matrix_power(inverse(A), -n)
    result = identity(A.shape[0])
    base = Matrix([list(row) if not isinstance(row, list) else row[:] for row in A.data])
    exponent = n
    while exponent > 0:
        if exponent % 2 == 1:
            result = matmul(result, base)
        base = matmul(base, base)
        exponent //= 2
    return result