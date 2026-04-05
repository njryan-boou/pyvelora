from __future__ import annotations

import cmath

from pyvelora.core import Matrix
from pyvelora.linalg.constructors import identity
from pyvelora.linalg.products import matmul


def ensure_square_matrix(A: Matrix) -> list[list[complex]]:
    if not isinstance(A, Matrix):
        raise TypeError("Input must be of type Matrix.")
    if A.shape[0] != A.shape[1]:
        raise ValueError("Input must be a square matrix.")
    return [[complex(value) for value in row] for row in A.data]


def copy_matrix(A: list[list[complex]]) -> list[list[complex]]:
    return [row[:] for row in A]


def trace(A: list[list[complex]]) -> complex:
    return sum(A[i][i] for i in range(len(A)))


def matrix_multiply(A: list[list[complex]], B: list[list[complex]]) -> list[list[complex]]:
    return [row[:] for row in matmul(Matrix(A), Matrix(B)).data]


def scale_identity(n: int, scalar: complex) -> list[list[complex]]:
    I = [[complex(value) for value in row] for row in identity(n).data]
    for i in range(n):
        I[i][i] *= scalar
    return I


def add(A: list[list[complex]], B: list[list[complex]]) -> list[list[complex]]:
    return [[a + b for a, b in zip(row_a, row_b)] for row_a, row_b in zip(A, B)]


def characteristic_polynomial_coeffs(A: list[list[complex]]) -> list[complex]:
    n = len(A)
    coeffs = [1.0 + 0j]
    B = [[complex(value) for value in row] for row in identity(n).data]

    for k in range(1, n + 1):
        AB = matrix_multiply(A, B)
        coeff = -trace(AB) / k
        coeffs.append(coeff)
        B = add(AB, scale_identity(n, coeff))

    return coeffs


def poly_eval(coeffs: list[complex], z: complex) -> complex:
    value = 0j
    for coeff in coeffs:
        value = value * z + coeff
    return value


def durand_kerner(coeffs: list[complex], max_iter: int = 200, tol: float = 1e-12) -> list[complex]:
    degree = len(coeffs) - 1
    if degree == 1:
        return [-coeffs[1] / coeffs[0]]

    radius = 1.0 + max(abs(coeff) for coeff in coeffs[1:])
    roots = [radius * cmath.exp(2j * cmath.pi * index / degree) for index in range(degree)]

    for _ in range(max_iter):
        next_roots: list[complex] = []
        converged = True

        for index, root in enumerate(roots):
            denominator = 1.0 + 0j
            for other_index, other_root in enumerate(roots):
                if index != other_index:
                    gap = root - other_root
                    if abs(gap) < tol:
                        gap = tol
                    denominator *= gap

            correction = poly_eval(coeffs, root) / denominator
            next_root = root - correction
            if abs(next_root - root) > tol:
                converged = False
            next_roots.append(next_root)

        roots = next_roots
        if converged:
            break

    return roots


def clean_scalar(value: complex, tol: float = 1e-10) -> float | complex:
    real = 0.0 if abs(value.real) < tol else value.real
    imag = 0.0 if abs(value.imag) < tol else value.imag
    if imag == 0.0:
        return float(real)
    return complex(real, imag)


def rref(A: list[list[complex]], tol: float = 1e-10) -> tuple[list[list[complex]], list[int]]:
    matrix = copy_matrix(A)
    rows = len(matrix)
    cols = len(matrix[0]) if rows else 0
    pivot_cols: list[int] = []
    pivot_row = 0

    for col in range(cols):
        if pivot_row >= rows:
            break

        best_row = max(range(pivot_row, rows), key=lambda row: abs(matrix[row][col]))
        if abs(matrix[best_row][col]) < tol:
            continue

        matrix[pivot_row], matrix[best_row] = matrix[best_row], matrix[pivot_row]
        pivot_value = matrix[pivot_row][col]
        matrix[pivot_row] = [entry / pivot_value for entry in matrix[pivot_row]]

        for row in range(rows):
            if row == pivot_row:
                continue
            factor = matrix[row][col]
            if abs(factor) < tol:
                continue
            matrix[row] = [entry - factor * pivot_entry for entry, pivot_entry in zip(matrix[row], matrix[pivot_row])]

        pivot_cols.append(col)
        pivot_row += 1

    for row in range(rows):
        for col in range(cols):
            if abs(matrix[row][col]) < tol:
                matrix[row][col] = 0j

    return matrix, pivot_cols


def null_vector(A: list[list[complex]], tol: float = 1e-10) -> list[complex]:
    reduced, pivot_cols = rref(A, tol=tol)
    cols = len(A[0]) if A else 0
    free_cols = [col for col in range(cols) if col not in pivot_cols]

    if not free_cols:
        free_cols = [cols - 1]

    vector = [0j for _ in range(cols)]
    vector[free_cols[0]] = 1.0 + 0j

    for row in range(len(pivot_cols) - 1, -1, -1):
        pivot_col = pivot_cols[row]
        total = 0j
        for col in range(pivot_col + 1, cols):
            total += reduced[row][col] * vector[col]
        vector[pivot_col] = -total

    norm = cmath.sqrt(sum(abs(value) ** 2 for value in vector))
    if abs(norm) >= tol:
        vector = [value / norm for value in vector]
    return vector