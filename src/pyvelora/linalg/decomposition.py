# Matrix decompositions and related utilities
from __future__ import annotations
import numpy as np

from ..core import Vector, Matrix

def lu_decomposition(A: Matrix) -> tuple[Matrix, Matrix]:
    """Return the LU decomposition of a square matrix A as (L, U)."""
    
    L = np.tril(A.data, k=-1) + np.eye(A.shape[0])
    U = np.triu(A.data)
    return Matrix(L), Matrix(U)

def qr_decomposition(A: Matrix) -> tuple[Matrix, Matrix]:
    """Return the QR decomposition of a square matrix A as (Q, R)."""
    Q, R = np.linalg.qr(A.data)
    return Matrix(Q), Matrix(R)

def svd_decomposition(A: Matrix) -> tuple[Matrix, Vector, Matrix]:
    """Return the SVD decomposition of a matrix A as (U, S, Vh)."""
    U, S, Vh = np.linalg.svd(A.data, full_matrices=False)
    return Matrix(U), Vector(S), Matrix(Vh)
