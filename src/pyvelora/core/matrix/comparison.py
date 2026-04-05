from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyvelora.core.matrix.matrix import Matrix


class MatrixComparison:
    """Comparison operations mixin for Matrix class."""

    def __eq__(self, other) -> Matrix:
        if not isinstance(other, type(self)):
            return NotImplemented
        return type(self)([[a == b for a, b in zip(row_self, row_other)] for row_self, row_other in zip(self.data, other.data)])

    def __ne__(self, other) -> Matrix:
        if not isinstance(other, type(self)):
            return NotImplemented
        return type(self)([[a != b for a, b in zip(row_self, row_other)] for row_self, row_other in zip(self.data, other.data)])

    def __lt__(self, other) -> Matrix:
        if not isinstance(other, type(self)):
            return NotImplemented
        return type(self)([[a < b for a, b in zip(row_self, row_other)] for row_self, row_other in zip(self.data, other.data)])

    def __le__(self, other) -> Matrix:
        if not isinstance(other, type(self)):
            return NotImplemented
        return type(self)([[a <= b for a, b in zip(row_self, row_other)] for row_self, row_other in zip(self.data, other.data)])

    def __gt__(self, other) -> Matrix:
        if not isinstance(other, type(self)):
            return NotImplemented
        return type(self)([[a > b for a, b in zip(row_self, row_other)] for row_self, row_other in zip(self.data, other.data)])

    def __ge__(self, other) -> Matrix:
        if not isinstance(other, type(self)):
            return NotImplemented
        return type(self)([[a >= b for a, b in zip(row_self, row_other)] for row_self, row_other in zip(self.data, other.data)])
