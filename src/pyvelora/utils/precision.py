from __future__ import annotations

import numpy as np

a_tol: float = 1e-12
r_tol: float = 1e-9
def isclose(a, b):
    """Elementwise closeness check within configured tolerance."""
    return np.isclose(a, b, atol=a_tol, rtol=r_tol)

def allclose(a: np.ndarray, b: np.ndarray) -> bool:
    """Check if all elements of two arrays are close within a tolerance."""
    return np.allclose(a, b, atol=a_tol, rtol=r_tol)

def zero_threshold(a: np.ndarray | float) -> np.ndarray | float:
    """Set values close to zero to exactly zero."""
    if np.isscalar(a):
        return 0.0 if np.abs(a) < a_tol else a
    a[np.abs(a) < a_tol] = 0
    return a

def clean(a: np.ndarray | float) -> np.ndarray | float:
    """Clean an array by applying zero thresholding."""
    return zero_threshold(a)

def set_precision(atol: float = 1e-12, rtol: float = 1e-9):
    """Set the precision for isclose and allclose functions."""
    global a_tol, r_tol
    a_tol = atol
    r_tol = rtol
    
def get_precision() -> tuple[float, float]:
    """Get the current precision settings."""
    return a_tol, r_tol