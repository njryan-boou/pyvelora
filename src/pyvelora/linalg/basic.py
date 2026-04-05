from __future__ import annotations
from pyvelora.core import Matrix, Vector, Tensor


def _as_row_list(row):
    if isinstance(row, list):
        return row[:]
    return list(row)


def _elementwise_binary(left, right, op):
    if isinstance(left, list) and isinstance(right, list):
        return [_elementwise_binary(l_item, r_item, op) for l_item, r_item in zip(left, right)]
    return op(left, right)


def _scalar_recursive(value, scalar):
    if isinstance(value, list):
        return [_scalar_recursive(item, scalar) for item in value]
    return value * scalar

def get_row(A: Matrix, i: int) -> Vector:
    """Get the i-th row of a matrix as a Vector."""
    if not isinstance(A, Matrix):
        raise TypeError("First argument must be a Matrix")
    if not isinstance(i, int) or i < 0 or i >= A.shape[0]:
        raise ValueError("Second argument must be a valid row index")
    return Vector(A.data[i])

def get_col(A: Matrix, j: int) -> Vector:
    """Get the j-th column of a matrix as a Vector."""
    if not isinstance(A, Matrix):
        raise TypeError("First argument must be a Matrix")
    if not isinstance(j, int) or j < 0 or j >= A.shape[1]:
        raise ValueError("Second argument must be a valid column index")
    return Vector([row[j] for row in A.data])

def swap_rows(A: Matrix, i: int, j: int) -> Matrix:
    """Return a new matrix with rows i and j swapped."""
    if not isinstance(A, Matrix):
        raise TypeError("First argument must be a Matrix")
    if not all(isinstance(idx, int) and 0 <= idx < A.shape[0] for idx in (i, j)):
        raise ValueError("Row indices must be valid")
    new_data = [_as_row_list(row) for row in A.data]
    new_data[i], new_data[j] = new_data[j], new_data[i]
    return Matrix(new_data)

def swap_cols(A: Matrix, i: int, j: int) -> Matrix:
    """Return a new matrix with columns i and j swapped."""
    if not isinstance(A, Matrix):
        raise TypeError("First argument must be a Matrix")
    if not all(isinstance(idx, int) and 0 <= idx < A.shape[1] for idx in (i, j)):
        raise ValueError("Column indices must be valid")
    new_data = [_as_row_list(row) for row in A.data]
    for row in new_data:
        row[i], row[j] = row[j], row[i]
    return Matrix(new_data)

def transpose(A: Matrix) -> Matrix:
    """Return the transpose of a matrix."""
    if not isinstance(A, Matrix):
        raise TypeError("Argument must be a Matrix")
    return Matrix([[A.data[row][col] for row in range(A.shape[0])] for col in range(A.shape[1])])

def add(A: Matrix | Vector | Tensor, B: Matrix | Vector | Tensor) -> Matrix | Vector | Tensor:
    """Return the sum of two matrices, vectors, or tensors."""
    if not isinstance(A, (Matrix, Vector, Tensor)) or not isinstance(B, (Matrix, Vector, Tensor)):
        raise TypeError("Arguments must be Matrices, Vectors, or Tensors")
    if A.shape != B.shape:
        raise ValueError("Matrices, Vectors, or Tensors must have the same shape for addition")
    if isinstance(A, Matrix) and isinstance(B, Matrix):
        return Matrix(_elementwise_binary(A.data, B.data, lambda left, right: left + right))
    elif isinstance(A, Vector) and isinstance(B, Vector):
        return Vector(_elementwise_binary(A.data, B.data, lambda left, right: left + right))
    else:
        return Tensor(_elementwise_binary(A.data, B.data, lambda left, right: left + right))
    
def subtract(A: Matrix | Vector | Tensor, B: Matrix | Vector | Tensor) -> Matrix | Vector | Tensor:
    """Return the difference of two matrices, vectors, or tensors."""
    if not isinstance(A, (Matrix, Vector, Tensor)) or not isinstance(B, (Matrix, Vector, Tensor)):
        raise TypeError("Arguments must be Matrices, Vectors, or Tensors")
    if A.shape != B.shape:
        raise ValueError("Matrices, Vectors, or Tensors must have the same shape for subtraction")
    if isinstance(A, Matrix) and isinstance(B, Matrix):
        return Matrix(_elementwise_binary(A.data, B.data, lambda left, right: left - right))
    elif isinstance(A, Vector) and isinstance(B, Vector):
        return Vector(_elementwise_binary(A.data, B.data, lambda left, right: left - right))
    else:
        return Tensor(_elementwise_binary(A.data, B.data, lambda left, right: left - right))
    
def scalar_multiply(A: Matrix | Vector | Tensor, scalar: float) -> Matrix | Vector | Tensor:
    """Return the product of a matrix, vector, or tensor with a scalar."""
    if not isinstance(A, (Matrix, Vector, Tensor)):
        raise TypeError("First argument must be a Matrix, Vector, or Tensor")
    if not isinstance(scalar, (int, float)):
        raise TypeError("Second argument must be a scalar")
    if isinstance(A, Matrix):
        return Matrix(_scalar_recursive(A.data, scalar))
    elif isinstance(A, Vector):
        return Vector(_scalar_recursive(A.data, scalar))
    else:
        return Tensor(_scalar_recursive(A.data, scalar))
    
def hadamard_product(A: Matrix, B: Matrix) -> Matrix:
    """Return the element-wise product of two matrices."""
    if not isinstance(A, Matrix) or not isinstance(B, Matrix):
        raise TypeError("Arguments must be Matrices")
    if A.shape != B.shape:
        raise ValueError("Matrices must have the same shape for Hadamard product")
    return Matrix(_elementwise_binary(A.data, B.data, lambda left, right: left * right))


def hamard_product(A: Matrix, B: Matrix) -> Matrix:
    """Backward-compatible alias for the historical misspelling."""
    return hadamard_product(A, B)