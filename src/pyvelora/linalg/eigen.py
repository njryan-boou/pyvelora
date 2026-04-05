from __future__ import annotations

from pyvelora.core import Matrix, Vector
from pyvelora.linalg._eigen_utils import characteristic_polynomial_coeffs
from pyvelora.linalg._eigen_utils import clean_scalar
from pyvelora.linalg._eigen_utils import copy_matrix
from pyvelora.linalg._eigen_utils import durand_kerner
from pyvelora.linalg._eigen_utils import ensure_square_matrix
from pyvelora.linalg._eigen_utils import null_vector


def eigenvalues(A: Matrix) -> Vector:
    """Compute the eigenvalues of a square matrix A without NumPy."""
    matrix = ensure_square_matrix(A)
    coeffs = characteristic_polynomial_coeffs(matrix)
    roots = durand_kerner(coeffs)
    return Vector([clean_scalar(root) for root in roots])


def eigenvectors(A: Matrix) -> Matrix:
    """Compute eigenvectors as columns of a matrix without NumPy."""
    matrix = ensure_square_matrix(A)
    n = len(matrix)
    values = eigenvalues(A).data
    vectors: list[list[complex]] = []

    for eigenvalue in values:
        shifted = copy_matrix(matrix)
        for index in range(n):
            shifted[index][index] -= complex(eigenvalue)
        vectors.append(null_vector(shifted))

    return Matrix([
        [clean_scalar(vectors[col][row]) for col in range(len(vectors))]
        for row in range(n)
    ])