from __future__ import annotations

import numpy as np

from ..core import Matrix, Vector


def transpose(matrix: Matrix) -> Matrix:
	"""Return the transpose of a matrix."""
	if not isinstance(matrix, Matrix):
		raise TypeError("Transpose is only defined for Matrix objects.")
	return matrix.transpose()


def determinant(matrix: Matrix) -> float:
	"""Return the determinant of a square matrix."""
	if not isinstance(matrix, Matrix):
		raise TypeError("Determinant is only defined for Matrix objects.")
	return matrix.determinant()


def inverse(matrix: Matrix) -> Matrix:
	"""Return the inverse of a square matrix."""
	if not isinstance(matrix, Matrix):
		raise TypeError("Inverse is only defined for Matrix objects.")
	return matrix.inverse()


def trace(matrix: Matrix) -> float:
	"""Return the trace of a square matrix."""
	if not isinstance(matrix, Matrix):
		raise TypeError("Trace is only defined for Matrix objects.")
	return matrix.trace()


def eigenvalues(matrix: Matrix) -> np.ndarray:
	"""Return the eigenvalues of a square matrix."""
	if not isinstance(matrix, Matrix):
		raise TypeError("Eigenvalues are only defined for Matrix objects.")
	return matrix.eigenvalues()


def eigenvectors(matrix: Matrix) -> tuple[Vector, Matrix]:
	"""Return the eigenvalues and eigenvectors of a square matrix."""
	if not isinstance(matrix, Matrix):
		raise TypeError("Eigenvectors are only defined for Matrix objects.")
	return matrix.eigenvectors()


def solve(matrix: Matrix, b: Vector) -> Vector:
	"""Solve the linear system Ax = b."""
	if not isinstance(matrix, Matrix):
		raise TypeError("Solve is only defined for Matrix objects.")
	return matrix.solve(b)


def matrix_power(matrix: Matrix, exponent: int | float | Matrix) -> Matrix:
	"""Raise a matrix to an exponent using Matrix.__pow__."""
	if not isinstance(matrix, Matrix):
		raise TypeError("Matrix power is only defined for Matrix objects.")
	result = matrix ** exponent
	if result is NotImplemented:
		raise TypeError("Unsupported exponent type for matrix power.")
	return result

