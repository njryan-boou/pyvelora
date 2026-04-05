from __future__ import annotations

class PyveloraError(Exception):
    """Base exception for all pyvelora errors."""
    pass


class ShapeError(PyveloraError):
    """Raised when array shape is invalid or incompatible."""
    pass


class DimensionError(PyveloraError):
    """Raised when dimensions are incompatible for an operation."""
    pass


class SingularMatrixError(ShapeError):
    """Raised when an operation requires an invertible matrix but the matrix is singular."""
    pass


class ConvergenceError(PyveloraError):
    """Raised when an iterative numerical method fails to converge."""
    pass


class DomainError(PyveloraError):
    """Raised when an input is outside the valid domain for an operation (e.g. log of a negative number)."""
    pass

