from __future__ import annotations

import math


def _infer_ndim(value) -> int:
    if hasattr(value, "ndim"):
        return int(value.ndim)
    current = getattr(value, "data", value)
    depth = 0
    while isinstance(current, list):
        depth += 1
        if not current:
            break
        current = current[0]
    return depth


def _flatten(value) -> list:
    """Recursively flatten a nested list (or object with .data) to a flat list."""
    raw = getattr(value, "data", value)
    result = []
    stack = [raw]
    while stack:
        item = stack.pop()
        if isinstance(item, list):
            stack.extend(reversed(item))
        else:
            result.append(item)
    return result


def require_vector(x, message: str = "Expected Vector"):
    from pyvelora.core.vector import Vector

    if not isinstance(x, Vector):
        raise TypeError(message)


def require_matrix(x, message: str = "Expected Matrix"):
    from pyvelora.core.matrix import Matrix

    if not isinstance(x, Matrix):
        raise TypeError(message)


def require_tensor(x, message: str = "Expected Tensor"):
    from pyvelora.core.tensor import Tensor

    if not isinstance(x, Tensor):
        raise TypeError(message)


def require_same_shape(a, b, message: str = "Shape mismatch"):
    if a.shape != b.shape:
        raise ValueError(message)


def require_square(A, message: str = "Matrix must be square"):
    if _infer_ndim(A) != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(message)


def require_dimension(x, dim: int, message: str | None = None):
    if _infer_ndim(x) != dim:
        raise ValueError(message or f"Expected dimension {dim}")


def isscalar(x) -> bool:
    return isinstance(x, (int, float, complex))


def require_scalar(x, message: str = "Expected scalar value"):
    if not isscalar(x):
        raise TypeError(message)


def require_nonzero(x, message: str = "Expected non-zero value"):
    if isscalar(x):
        if x == 0:
            raise ValueError(message)
        return
    if any(v == 0 for v in _flatten(x)):
        raise ValueError(message)


def require_positive(x, message: str = "Expected positive value"):
    if isscalar(x):
        if x <= 0:
            raise ValueError(message)
        return
    if any(v <= 0 for v in _flatten(x)):
        raise ValueError(message)


def require_nonnegative(x, message: str = "Expected non-negative value"):
    if isscalar(x):
        if x < 0:
            raise ValueError(message)
        return
    if any(v < 0 for v in _flatten(x)):
        raise ValueError(message)


def require_integer(x, message: str = "Expected integer value"):
    if isscalar(x):
        if not float(x.real if isinstance(x, complex) else x).is_integer():
            raise ValueError(message)
        return
    if any(not float(v.real if isinstance(v, complex) else v).is_integer() for v in _flatten(x)):
        raise ValueError(message)


def require_real(x, message: str = "Expected real value"):
    if isscalar(x):
        if isinstance(x, complex) and x.imag != 0:
            raise ValueError(message)
        return
    if any(isinstance(v, complex) and v.imag != 0 for v in _flatten(x)):
        raise ValueError(message)


def require_complex(x, message: str = "Expected complex value"):
    if isscalar(x):
        if not isinstance(x, complex):
            raise ValueError(message)
        return
    if all(not isinstance(v, complex) for v in _flatten(x)):
        raise ValueError(message)


def require_finite(x, message: str = "Expected finite value"):
    if isscalar(x):
        val = x.real if isinstance(x, complex) else x
        if not math.isfinite(val):
            raise ValueError(message)
        return
    if any(not math.isfinite(v.real if isinstance(v, complex) else v) for v in _flatten(x)):
        raise ValueError(message)




