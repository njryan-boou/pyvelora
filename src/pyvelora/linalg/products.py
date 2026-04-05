from __future__ import annotations

from pyvelora.core import Vector, Matrix

def dot(u: Vector, v: Vector) -> float:
    """Compute the dot product of two vectors."""
    if not isinstance(u, Vector) or not isinstance(v, Vector) or u.shape[0] != v.shape[0]:
        raise TypeError("Both arguments must be Vectors of the same length")
    return float(sum(x * y for x, y in zip(u.data, v.data)))

def outer(u: Vector, v: Vector) -> Matrix:
    """Compute the outer product of two vectors."""
    if not isinstance(u, Vector) or not isinstance(v, Vector):
        raise TypeError("Both arguments must be Vectors")
    return Matrix([[x * y for y in v.data] for x in u.data])

def matmul(A: Matrix, B: Matrix) -> Matrix:
    """Compute the matrix product of A and B."""
    if not isinstance(A, Matrix) or not isinstance(B, Matrix):
        raise TypeError("Both arguments must be Matrices")
    if A.shape[1] != B.shape[0]:
        raise ValueError("Inner dimensions must match for matrix multiplication")
    return Matrix([[sum(a * b for a, b in zip(row, col)) for col in zip(*B.data)] for row in A.data])

def matvec(A: Matrix, v: Vector) -> Vector:
    """Compute the matrix-vector product of A and v."""
    if not isinstance(A, Matrix) or not isinstance(v, Vector):
        raise TypeError("First argument must be a Matrix and second argument must be a Vector")
    if A.shape[1] != v.shape[0]:
        raise ValueError("Inner dimensions must match for matrix-vector multiplication")
    return Vector([sum(a * b for a, b in zip(row, v.data)) for row in A.data])

def cross(u: Vector, v: Vector) -> Vector:
    """Compute the cross product of two 3D vectors."""
    if not isinstance(u, Vector) or not isinstance(v, Vector):
        raise TypeError("Both arguments must be Vectors")
    if u.shape[0] != 3 or v.shape[0] != 3:
        raise ValueError("Both vectors must be 3D")
    result = u.data[1] * v.data[2] - u.data[2] * v.data[1], \
             u.data[2] * v.data[0] - u.data[0] * v.data[2], \
             u.data[0] * v.data[1] - u.data[1] * v.data[0]
    return Vector(list(result))

def prod(A: Matrix, axis: int = 0) -> Vector:
    """Compute the product of elements along a given axis of a matrix."""
    if not isinstance(A, Matrix):
        raise TypeError("First argument must be a Matrix")
    if axis not in (0, 1):
        raise ValueError("Axis must be 0 (columns) or 1 (rows)")
    if axis == 1:
        return Vector([_product(row) for row in A.data])
    return Vector([_product(col) for col in zip(*A.data)])


def _product(values) -> float:
    result = 1
    for value in values:
        result *= value
    return result