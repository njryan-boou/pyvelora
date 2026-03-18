from __future__ import annotations
import numpy as np
from ...core import Matrix

def second_order(f):
    """
    Convert y'' = f(t, y, y') into first order system.
    """

    def system(t, Y):
        y, v = Y
        return [v, f(t, y, v)]

    return system


def linear(A):
    """
    Create system x' = A x
    """

    A = Matrix(A)

    def system(t, x):
        return A @ x

    return system