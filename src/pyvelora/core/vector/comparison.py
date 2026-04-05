from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyvelora.core.vector.vector import Vector
    from pyvelora.core.matrix import Matrix

# Comparison operations for Vector class
class VectorComparison:
    """Comparison operations mixin for Vector class."""

    def __eq__(self, other: Vector) -> bool:
        """Check if two vectors are equal."""
        if not isinstance(other, type(self)):
            return NotImplemented
        if self.size != other.size:
            return False
        return all(a == b for a, b in zip(self.data, other.data))

    def __ne__(self, other: Vector) -> bool:
        """Check if two vectors are not equal."""
        if not isinstance(other, type(self)):
            return NotImplemented
        if self.size != other.size:
            return True
        for a, b in zip(self.data, other.data):
            if a != b:
                return True
        return False

    def __lt__(self, other: Vector) -> bool:
        """Check if vector a is less than vector b (element-wise)."""
        if not isinstance(other, type(self)):
            return NotImplemented
        if self.size != other.size:
            raise ValueError("Vectors must be of the same size for comparison.")
        return all(a < b for a, b in zip(self.data, other.data))

    def __le__(self, other: Vector) -> bool:
        """Check if vector a is less than or equal to vector b (element-wise)."""
        if not isinstance(other, type(self)):
            return NotImplemented
        if self.size != other.size:
            raise ValueError("Vectors must be of the same size for comparison.")
        return all(a <= b for a, b in zip(self.data, other.data))

    def __gt__(self, other: Vector) -> bool:
        """Check if vector a is greater than vector b (element-wise)."""
        if not isinstance(other, type(self)):
            return NotImplemented
        if self.size != other.size:
            raise ValueError("Vectors must be of the same size for comparison.")
        return all(a > b for a, b in zip(self.data, other.data))

    def __ge__(self, other: Vector) -> bool:
        """Check if vector a is greater than or equal to vector b (element-wise)."""
        if not isinstance(other, type(self)):
            return NotImplemented
        if self.size != other.size:
            raise ValueError("Vectors must be of the same size for comparison.")
        return all(a >= b for a, b in zip(self.data, other.data))