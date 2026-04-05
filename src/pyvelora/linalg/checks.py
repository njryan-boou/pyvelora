from __future__ import annotations
from pyvelora.core import Vector, Matrix
from pyvelora.linalg.basic import transpose
from pyvelora.linalg.constructors import diagonal, identity
from pyvelora.linalg.properties import determinant
from pyvelora.linalg.products import matmul


def _matrices_close(A: list[list[float | complex]], B: list[list[float | complex]], tol: float = 1e-10) -> bool:
    if len(A) != len(B):
        return False
    return all(
        len(row_a) == len(row_b) and all(abs(left - right) <= tol for left, right in zip(row_a, row_b))
        for row_a, row_b in zip(A, B)
    )
def is_square(A: Matrix) -> bool:
    """Check if a matrix is square."""
    if not isinstance(A, Matrix):
        raise TypeError("Argument must be a Matrix")
    return A.shape[0] == A.shape[1]

def is_symmetric(A: Matrix) -> bool:
    """Check if a matrix is symmetric."""
    if not isinstance(A, Matrix):
        raise TypeError("Argument must be a Matrix")
    if not is_square(A):
        return False
    return _matrices_close(A.data, transpose(A).data)

def is_orthogonal(A: Matrix) -> bool:
    """Check if a matrix is orthogonal."""
    if not isinstance(A, Matrix):
        raise TypeError("Argument must be a Matrix")
    if not is_square(A):
        return False
    return _matrices_close(matmul(transpose(A), A).data, identity(A.shape[0]).data)

def is_singular(A: Matrix) -> bool:
    """Check if a matrix is singular."""
    if not isinstance(A, Matrix):
        raise TypeError("Argument must be a Matrix")
    if not is_square(A):
        return False
    return determinant(A) == 0

def is_invertible(A: Matrix) -> bool:
    """Check if a matrix is invertible."""
    if not isinstance(A, Matrix):
        raise TypeError("Argument must be a Matrix")
    if not is_square(A):
        return False
    return not is_singular(A)

def is_diagonal(A: Matrix) -> bool:
    """Check if a matrix is diagonal."""
    if not isinstance(A, Matrix):
        raise TypeError("Argument must be a Matrix")
    if not is_square(A):
        return False
    for row in range(A.shape[0]):
        for col in range(A.shape[1]):
            if row != col and abs(A.data[row][col]) > 1e-10:
                return False
    return True

def is_identity(A: Matrix) -> bool:
    """Check if a matrix is the identity matrix."""
    if not isinstance(A, Matrix):
        raise TypeError("Argument must be a Matrix")
    if not is_square(A):
        return False
    return _matrices_close(A.data, identity(A.shape[0]).data)

def is_upper_triangular(A: Matrix) -> bool:
    """Check if a matrix is upper triangular."""
    if not isinstance(A, Matrix):
        raise TypeError("Argument must be a Matrix")
    if not is_square(A):
        return False
    for row in range(1, A.shape[0]):
        for col in range(min(row, A.shape[1])):
            if abs(A.data[row][col]) > 1e-10:
                return False
    return True

def is_lower_triangular(A: Matrix) -> bool:
    """Check if a matrix is lower triangular."""
    if not isinstance(A, Matrix):
        raise TypeError("Argument must be a Matrix")
    if not is_square(A):
        return False
    for row in range(A.shape[0]):
        for col in range(row + 1, A.shape[1]):
            if abs(A.data[row][col]) > 1e-10:
                return False
    return True

def is_rref(A: Matrix) -> bool:
    """Check if a matrix is in reduced row echelon form."""
    if not isinstance(A, Matrix):
        raise TypeError("Argument must be a Matrix")
    rows, cols = A.shape
    lead = 0
    for r in range(rows):
        if lead >= cols:
            return True
        i = r
        while A.data[i][lead] == 0:
            i += 1
            if i == rows:
                i = r
                lead += 1
                if lead == cols:
                    return True
        if i != r:
            return False  # Not in RREF if we had to swap rows
        # Check for leading 1 and zeros below it
        if A.data[r][lead] != 1:
            return False  # Leading entry is not 1
        for j in range(r + 1, rows):
            if A.data[j][lead] != 0:
                return False  # Non-zero entry below leading 1
        lead += 1
    return True


def is_skew_symmetric(A: Matrix) -> bool:
    """Check if a matrix is skew-symmetric (A^T = -A)."""
    if not isinstance(A, Matrix):
        raise TypeError("Argument must be a Matrix")
    if not is_square(A):
        return False
    for row in range(A.shape[0]):
        for col in range(A.shape[1]):
            if abs(A.data[row][col] + A.data[col][row]) > 1e-10:
                return False
    return True


def is_positive_definite(A: Matrix) -> bool:
    """Check if a matrix is positive definite using Sylvester's criterion."""
    if not isinstance(A, Matrix):
        raise TypeError("Argument must be a Matrix")
    if not is_square(A):
        return False
    if not is_symmetric(A):
        return False
    n = A.shape[0]
    for k in range(1, n + 1):
        leading_minor = Matrix([row[:k] for row in A.data[:k]])
        if determinant(leading_minor) <= 0:
            return False
    return True