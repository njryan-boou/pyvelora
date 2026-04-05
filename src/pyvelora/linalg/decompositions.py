from __future__ import annotations
from math import atan2, cos, sin, sqrt

from pyvelora.core import Matrix, Vector
from pyvelora.linalg.basic import get_col, transpose
from pyvelora.linalg.constructors import diagonal, zeros
from pyvelora.linalg.norms import vector_norm
from pyvelora.linalg.products import matmul
from pyvelora.linalg.products import dot

def lu_decomposition(A: Matrix) -> tuple[Matrix, Matrix]:
    """Perform LU decomposition of a square matrix A."""
    if not isinstance(A, Matrix):
        raise TypeError("Argument must be a Matrix")
    if A.shape[0] != A.shape[1]:
        raise ValueError("LU decomposition is only defined for square matrices")
    L = zeros(A.shape[0], A.shape[1])
    U = zeros(A.shape[0], A.shape[1])
    n = A.shape[0]
    for i in range(n):
        L.data[i][i] = 1.0
    for i in range(n):
        for j in range(i, n):
            U.data[i][j] = A.data[i][j]
            for k in range(i):
                U.data[i][j] -= L.data[i][k] * U.data[k][j]
        for j in range(i + 1, n):
            L.data[j][i] = A.data[j][i]
            for k in range(i):
                L.data[j][i] -= L.data[j][k] * U.data[k][i]
            L.data[j][i] /= U.data[i][i]
    return L, U

def qr_decomposition(A: Matrix) -> tuple[Matrix, Matrix]:
    """Perform QR decomposition of a matrix A."""
    if not isinstance(A, Matrix):
        raise TypeError("Argument must be a Matrix")
    m, n = A.shape
    q_columns: list[list[float]] = []
    R = zeros(n, n)
    for j in range(n):
        v = get_col(A, j).data[:]
        for i in range(j):
            R.data[i][j] = dot(Vector(q_columns[i]), get_col(A, j))
            v = [value - R.data[i][j] * basis for value, basis in zip(v, q_columns[i])]
        R.data[j][j] = vector_norm(Vector(v))
        if R.data[j][j] == 0:
            raise ValueError("Matrix is rank deficient")
        q_columns.append([value / R.data[j][j] for value in v])
    Q = Matrix([[q_columns[col][row] for col in range(n)] for row in range(m)])
    return Q, R

def svd_decomposition(A: Matrix) -> tuple[Matrix, Matrix, Matrix]:
    """Perform Singular Value Decomposition of a matrix A."""
    if not isinstance(A, Matrix):
        raise TypeError("Argument must be a Matrix")
    if A.shape != (2, 2):
        raise NotImplementedError("Pure-Python SVD currently supports only 2x2 matrices")

    a, b = float(A.data[0][0]), float(A.data[0][1])
    c, d = float(A.data[1][0]), float(A.data[1][1])

    ata00 = a * a + c * c
    ata01 = a * b + c * d
    ata11 = b * b + d * d
    tr = ata00 + ata11
    det = ata00 * ata11 - ata01 * ata01
    disc = max(0.0, tr * tr - 4.0 * det)
    root = sqrt(disc)

    l1 = 0.5 * (tr + root)
    l2 = 0.5 * (tr - root)
    s1 = sqrt(max(0.0, l1))
    s2 = sqrt(max(0.0, l2))

    if abs(ata01) > 1e-12:
        v1 = [l1 - ata11, ata01]
        v2 = [l2 - ata11, ata01]
    else:
        if ata00 >= ata11:
            v1, v2 = [1.0, 0.0], [0.0, 1.0]
        else:
            v1, v2 = [0.0, 1.0], [1.0, 0.0]

    n1 = sqrt(v1[0] * v1[0] + v1[1] * v1[1])
    n2 = sqrt(v2[0] * v2[0] + v2[1] * v2[1])
    v1 = [v1[0] / n1, v1[1] / n1]
    v2 = [v2[0] / n2, v2[1] / n2]
    V = [[v1[0], v2[0]], [v1[1], v2[1]]]

    if s1 > 1e-12:
        u1 = [(a * v1[0] + b * v1[1]) / s1, (c * v1[0] + d * v1[1]) / s1]
    else:
        u1 = [1.0, 0.0]
    if s2 > 1e-12:
        u2 = [(a * v2[0] + b * v2[1]) / s2, (c * v2[0] + d * v2[1]) / s2]
    else:
        u2 = [-u1[1], u1[0]]

    # Re-orthonormalize U for numerical stability
    u1n = sqrt(u1[0] * u1[0] + u1[1] * u1[1])
    u1 = [u1[0] / u1n, u1[1] / u1n]
    proj = u2[0] * u1[0] + u2[1] * u1[1]
    u2 = [u2[0] - proj * u1[0], u2[1] - proj * u1[1]]
    u2n = sqrt(u2[0] * u2[0] + u2[1] * u2[1])
    if u2n < 1e-12:
        u2 = [-u1[1], u1[0]]
    else:
        u2 = [u2[0] / u2n, u2[1] / u2n]

    U = Matrix([[u1[0], u2[0]], [u1[1], u2[1]]])
    S = Matrix([[s1, 0.0], [0.0, s2]])
    Vh = Matrix([[V[0][0], V[1][0]], [V[0][1], V[1][1]]])
    return U, S, Vh


def polar_decomposition(A: Matrix) -> tuple[Matrix, Matrix]:
    """Perform polar decomposition of a matrix A."""
    if not isinstance(A, Matrix):
        raise TypeError("Argument must be a Matrix")
    U_mat, S_mat, Vh_mat = svd_decomposition(A)
    # Positive semidefinite factor: P = V S V^T
    V = Matrix([[Vh_mat.data[0][0], Vh_mat.data[1][0]], [Vh_mat.data[0][1], Vh_mat.data[1][1]]])
    P = matmul(matmul(V, S_mat), Vh_mat)
    # Orthogonal factor: U = U_svd V^T
    U = matmul(U_mat, Vh_mat)
    return P, U


def eigen_decomposition(A: Matrix) -> tuple[Vector, Matrix]:
    """Perform eigendecomposition of a square matrix A.
    
    Returns (P, D) where A = P @ D @ inv(P), with D containing eigenvalues on the diagonal.
    """
    if not isinstance(A, Matrix):
        raise TypeError("Argument must be a Matrix")
    if A.shape[0] != A.shape[1]:
        raise ValueError("Eigendecomposition is only defined for square matrices")
    if A.shape != (2, 2):
        raise NotImplementedError("Pure-Python eigendecomposition currently supports only 2x2 matrices")

    a, b = float(A.data[0][0]), float(A.data[0][1])
    c, d = float(A.data[1][0]), float(A.data[1][1])
    tr = a + d
    det = a * d - b * c
    disc = tr * tr - 4.0 * det
    if disc < 0:
        raise ValueError("Complex eigenvalues are not supported in this implementation")
    root = sqrt(disc)
    l1 = 0.5 * (tr + root)
    l2 = 0.5 * (tr - root)

    def eigenvector(lam):
        if abs(b) > 1e-12:
            v = [b, lam - a]
        elif abs(c) > 1e-12:
            v = [lam - d, c]
        else:
            v = [1.0, 0.0]
        n = sqrt(v[0] * v[0] + v[1] * v[1])
        if n < 1e-12:
            return [1.0, 0.0]
        return [v[0] / n, v[1] / n]

    v1 = eigenvector(l1)
    v2 = eigenvector(l2)
    vals = Vector([l1, l2])
    vecs = Matrix([[v1[0], v2[0]], [v1[1], v2[1]]])
    return vals, vecs


def cholesky_decomposition(A: Matrix) -> Matrix:
    """Perform Cholesky decomposition of a symmetric positive-definite matrix A.
    
    Returns L such that A = L @ L.T
    """
    if not isinstance(A, Matrix):
        raise TypeError("Argument must be a Matrix")
    if A.shape[0] != A.shape[1]:
        raise ValueError("Cholesky decomposition is only defined for square matrices")
    n = A.shape[0]
    L = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                value = A.data[i][i] - s
                if value <= 0:
                    raise ValueError("Matrix must be symmetric positive definite")
                L[i][j] = sqrt(value)
            else:
                if abs(L[j][j]) < 1e-12:
                    raise ValueError("Matrix is not positive definite")
                L[i][j] = (A.data[i][j] - s) / L[j][j]
    return Matrix(L)


def schur_decomposition(A: Matrix) -> tuple[Matrix, Matrix]:
    """Perform Schur decomposition of a square matrix A.
    
    Returns (Q, T) where A = Q @ T @ Q.H, with Q unitary and T upper triangular.
    """
    if not isinstance(A, Matrix):
        raise TypeError("Argument must be a Matrix")
    if A.shape[0] != A.shape[1]:
        raise ValueError("Schur decomposition is only defined for square matrices")
    if A.shape != (2, 2):
        raise NotImplementedError("Pure-Python Schur decomposition currently supports only 2x2 matrices")

    a, b = float(A.data[0][0]), float(A.data[0][1])
    c, d = float(A.data[1][0]), float(A.data[1][1])
    if abs(c) < 1e-12:
        T = Matrix([[a, b], [0.0, d]])
        Z = Matrix([[1.0, 0.0], [0.0, 1.0]])
        return T, Z

    theta = 0.5 * atan2(2.0 * c, d - a)
    ct, st = cos(theta), sin(theta)
    Z = Matrix([[ct, -st], [st, ct]])
    zt = Matrix([[ct, st], [-st, ct]])
    T = matmul(matmul(zt, A), Z)
    T.data[1][0] = 0.0
    return T, Z
