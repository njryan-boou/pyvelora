from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyvelora.core.matrix.matrix import Matrix
    from pyvelora.core.vector.vector import Vector


class MatrixIndexing:
    """Indexing operations mixin for Matrix class."""

    def __iter__(self):
        for i in range(self.shape[0]):
            yield self.data[i]

    def __contains__(self, item) -> bool:
        return any(item in row for row in self.data)

    def __getitem__(self, key):
        from pyvelora.core.matrix.matrix import Matrix
        from pyvelora.core.vector.vector import Vector

        result = self.data[key]
        if isinstance(result, list):
            if result and isinstance(result[0], list):
                return Matrix(result)
            return Vector(result)
        return result

    def __setitem__(self, key, value) -> None:
        if isinstance(key, tuple):
            row_key, col_key = key
            self.data[row_key][col_key] = value
        else:
            self.data[key] = value

    def __len__(self):
        return len(self.data)
