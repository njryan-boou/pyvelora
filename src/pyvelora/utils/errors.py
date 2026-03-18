"""
Custom exceptions for the pyvelora library.

This module defines general exceptions for vector, matrix, and tensor operations.
"""


class PyveloraError(Exception):
    """Base exception for all pyvelora errors."""
    pass


class ShapeError(PyveloraError):
    """Raised when array shape is invalid or incompatible."""
    pass


class DimensionError(PyveloraError):
    """Raised when dimensions are incompatible for an operation."""
    pass