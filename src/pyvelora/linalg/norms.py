from __future__ import annotations
from pyvelora.core import Vector, Matrix

def vector_norm(v: Vector, p: int | float = 2) -> float:
    """Compute the p-norm of a vector (default Euclidean norm)."""
    if not isinstance(v, Vector):
        raise TypeError("Argument must be a Vector")
    if p == 1:
        return float(sum(abs(x) for x in v.data))
    if p == 2:
        return float(sum(x**2 for x in v.data) ** 0.5)
    if p == float("inf"):
        return float(max(abs(x) for x in v.data))
    raise ValueError("Only p=1, p=2, and p=inf are supported")

def frobenius_norm(A: Matrix) -> float:
    """Compute the Frobenius norm of a matrix."""
    if not isinstance(A, Matrix):
        raise TypeError("Argument must be a Matrix")
    return float(sum(value**2 for row in A.data for value in row) ** 0.5)

def one_norm(A: Matrix) -> float:
    """Compute the 1-norm of a matrix."""
    if not isinstance(A, Matrix):
        raise TypeError("Argument must be a Matrix")
    return float(max(sum(abs(value) for value in col) for col in zip(*A.data)))

def inf_norm(A: Matrix) -> float:
    """Compute the infinity norm of a matrix."""
    if not isinstance(A, Matrix):
        raise TypeError("Argument must be a Matrix")
    return float(max(sum(abs(value) for value in row) for row in A.data))

def normalize(v: Vector) -> Vector:
    """Return a normalized version of the input vector."""
    if not isinstance(v, Vector):
        raise TypeError("Argument must be a Vector")
    norm = vector_norm(v)
    if norm == 0.0:
        raise ValueError("Cannot normalize a zero vector")
    return Vector([x / norm for x in v.data])