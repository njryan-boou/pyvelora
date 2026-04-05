from pyvelora.core import Vector, Matrix
from pyvelora.linalg.basic import transpose
from pyvelora.linalg.rref import rref


def ColumnSpace(matrix: Matrix) -> list[Vector]:
    """Return the column space of the given matrix as a list of vectors."""
    _, pivot_cols = rref(matrix)
    return [Vector([row[col] for row in matrix.data]) for col in pivot_cols]


def RowSpace(matrix: Matrix) -> list[Vector]:
    """Return the row space of the given matrix as a list of vectors."""
    return [Vector(row) for row in matrix.data]

def NullSpace(matrix: Matrix) -> list[Vector]:
    """Return the null space of the given matrix as a list of vectors."""
    reduced, pivot_cols = rref(matrix)
    cols = matrix.shape[1]
    free_cols = [col for col in range(cols) if col not in pivot_cols]
    basis = []
    for free_col in free_cols:
        values = [0.0 for _ in range(cols)]
        values[free_col] = 1.0
        for row in range(len(pivot_cols) - 1, -1, -1):
            pivot_col = pivot_cols[row]
            total = 0.0
            for col in range(pivot_col + 1, cols):
                total += reduced.data[row][col] * values[col]
            values[pivot_col] = -total
        basis.append(Vector(values))
    return basis

def LeftNullSpace(matrix: Matrix) -> list[Vector]:
    """Return the left null space of the given matrix as a list of vectors."""
    return NullSpace(transpose(matrix))

if __name__ == "__main__":
    A = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print("Column Space:", ColumnSpace(A))
    print("Row Space:", RowSpace(A))
    print("Null Space:", NullSpace(A))
    print("Left Null Space:", LeftNullSpace(A))
    print(A)

# Lowercase aliases for consistent linalg API
column_space = ColumnSpace
row_space = RowSpace
null_space = NullSpace
left_null_space = LeftNullSpace