# solver
from __future__ import annotations

import numpy as np

from .matrix_ops import inverse
from ..core import Matrix, Vector
from .decomposition import lu_decomposition, qr_decomposition, svd_decomposition

def solve_lu(A: Matrix, b: Vector) -> Vector:
    """Solve the linear system Ax = b for x using LU decomposition."""
    L, U = lu_decomposition(A)
    
    # Forward substitution to solve Ly = b
    y = np.zeros_like(b.data)
    for i in range(len(L)):
        y[i] = (b.data[i] - np.dot(L.data[i, :i], y[:i])) / L.data[i, i]
    
    # Backward substitution to solve Ux = y
    x = np.zeros_like(y)
    for i in range(len(U) - 1, -1, -1):
        x[i] = (y[i] - np.dot(U.data[i, i + 1:], x[i + 1:])) / U.data[i, i]
    
    return Vector(x)

def solve_qr(A: Matrix, b: Vector) -> Vector:
    """Solve the linear system Ax = b for x using QR decomposition."""
    Q, R = qr_decomposition(A)
    y = Q.T @ b
    x = np.zeros_like(y.data)
    for i in range(len(R) - 1, -1, -1):
        x[i] = (y.data[i] - np.dot(R.data[i, i + 1:], x[i + 1:])) / R.data[i, i]
    return Vector(x)

def solve_svd(A: Matrix, b: Vector) -> Vector:
    """Solve the linear system Ax = b for x using SVD decomposition."""
    U, S, Vh = svd_decomposition(A)
    c = U.T @ b
    w = np.zeros_like(S)
    for i in range(len(S)):
        if S[i] > 1e-10:  # Avoid division by zero
            w[i] = c.data[i] / S[i]
    x = Vh.T @ w
    return Vector(x)

def solve_inverse(A: Matrix, b: Vector) -> Vector:
    """Solve the linear system Ax = b for x using matrix inverse."""
    A_inv = inverse(A)
    return A_inv @ b

def solve(A: Matrix, b: Vector, method: str = "lu") -> Vector:
    """Solve the linear system Ax = b for x using the specified method."""
    if method == "lu":
        return solve_lu(A, b)
    elif method == "qr":
        return solve_qr(A, b)
    elif method == "svd":
        return solve_svd(A, b)
    elif method == "inverse":

        return solve_inverse(A, b)
    else:
        raise ValueError(f"Unknown method '{method}'. Supported methods: lu, qr, svd, inverse.")

