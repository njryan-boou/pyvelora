from __future__ import annotations


from pyvelora.core import Matrix


def rref(matrix: Matrix) -> tuple[Matrix, tuple]:
    """Compute the reduced row echelon form of a Matrix and return the pivot columns."""

    if not isinstance(matrix, Matrix):
        raise TypeError("Input must be of type Matrix.")

    A = [[float(value) for value in row] for row in matrix.data]
    rows, cols = matrix.shape
    pivot_cols = []
    row = 0
    tolerance = 1e-12

    for col in range(cols):
        if row >= rows:
            break

        # Find the pivot in the current column.
        pivot_row = None
        for r in range(row, rows):
            if abs(A[r][col]) > tolerance:
                pivot_row = r
                break
        if pivot_row is None:
            continue

        A[row], A[pivot_row] = A[pivot_row], A[row]
        pivot_cols.append(col)

        pivot_value = A[row][col]
        A[row] = [value / pivot_value for value in A[row]]

        for r in range(row + 1, rows):
            factor = A[r][col]
            if abs(factor) <= tolerance:
                continue
            A[r] = [value - factor * pivot for value, pivot in zip(A[r], A[row])]

        row += 1

    for i in reversed(range(len(pivot_cols))):
        col = pivot_cols[i]
        for r in range(i):
            factor = A[r][col]
            if abs(factor) <= tolerance:
                continue
            A[r] = [value - factor * pivot for value, pivot in zip(A[r], A[i])]

    for r in range(rows):
        for c in range(cols):
            if abs(A[r][c]) <= tolerance:
                A[r][c] = 0.0

    return Matrix(A), tuple(pivot_cols)
