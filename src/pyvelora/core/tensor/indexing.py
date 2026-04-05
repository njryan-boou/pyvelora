from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyvelora.core.tensor.tensor import Tensor


class TensorIndexing:
    """Indexing operations mixin for Tensor class."""

    def __iter__(self):
        for i in range(self.shape[0]):
            yield self.data[i]

    def __contains__(self, item) -> bool:
        stack = [self.data]
        while stack:
            current = stack.pop()
            if isinstance(current, list):
                for value in current:
                    stack.append(value)
            elif isinstance(current, complex) or isinstance(item, complex):
                if abs(current - item) <= 1e-12:
                    return True
            elif abs(float(current) - float(item)) <= 1e-12:
                return True
        return False

    def __getitem__(self, key):
        from pyvelora.core.tensor.tensor import Tensor
        from pyvelora.core.matrix.matrix import Matrix

        def _apply_index(data, keys):
            """Recursively apply a tuple of index/slice keys to a nested list."""
            if not keys:
                return data
            k = keys[0]
            rest = keys[1:]
            if isinstance(k, int):
                return _apply_index(data[k], rest)
            # slice
            selected = list(data[k])
            if rest:
                return [_apply_index(item, rest) for item in selected]
            return selected

        if isinstance(key, tuple):
            result = _apply_index(self.data, key)
        else:
            result = self.data[key]

        if not isinstance(result, list):
            return result

        # Determine depth of result
        depth = 0
        current = result
        while isinstance(current, list):
            depth += 1
            if not current:
                break
            current = current[0]

        if depth >= 3:
            return Tensor(result)
        if depth == 2:
            return Matrix(result)
        return result

    def __setitem__(self, key, value) -> None:
        if not isinstance(key, tuple):
            self.data[key] = value
            return
        target = self.data
        for part in key[:-1]:
            target = target[part]
        target[key[-1]] = value

    def __len__(self) -> int:
        return len(self.data)
