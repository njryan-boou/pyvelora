from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyvelora.core.tensor.tensor import Tensor


class TensorComparison:
    """Comparison operations mixin for Tensor class."""

    def __eq__(self, other) -> Tensor:
        if not isinstance(other, type(self)):
            return NotImplemented
        result = []
        stack = [(self.data, other.data, result)]
        while stack:
            left, right, target = stack.pop()
            for left_item, right_item in zip(left, right):
                if isinstance(left_item, list):
                    nested = []
                    target.append(nested)
                    stack.append((left_item, right_item, nested))
                else:
                    target.append(left_item == right_item)
        return type(self)(result)

    def __ne__(self, other) -> Tensor:
        if not isinstance(other, type(self)):
            return NotImplemented
        result = []
        stack = [(self.data, other.data, result)]
        while stack:
            left, right, target = stack.pop()
            for left_item, right_item in zip(left, right):
                if isinstance(left_item, list):
                    nested = []
                    target.append(nested)
                    stack.append((left_item, right_item, nested))
                else:
                    target.append(left_item != right_item)
        return type(self)(result)

    def __lt__(self, other) -> Tensor:
        if not isinstance(other, type(self)):
            return NotImplemented
        result = []
        stack = [(self.data, other.data, result)]
        while stack:
            left, right, target = stack.pop()
            for left_item, right_item in zip(left, right):
                if isinstance(left_item, list):
                    nested = []
                    target.append(nested)
                    stack.append((left_item, right_item, nested))
                else:
                    target.append(left_item < right_item)
        return type(self)(result)

    def __le__(self, other) -> Tensor:
        if not isinstance(other, type(self)):
            return NotImplemented
        result = []
        stack = [(self.data, other.data, result)]
        while stack:
            left, right, target = stack.pop()
            for left_item, right_item in zip(left, right):
                if isinstance(left_item, list):
                    nested = []
                    target.append(nested)
                    stack.append((left_item, right_item, nested))
                else:
                    target.append(left_item <= right_item)
        return type(self)(result)

    def __gt__(self, other) -> Tensor:
        if not isinstance(other, type(self)):
            return NotImplemented
        result = []
        stack = [(self.data, other.data, result)]
        while stack:
            left, right, target = stack.pop()
            for left_item, right_item in zip(left, right):
                if isinstance(left_item, list):
                    nested = []
                    target.append(nested)
                    stack.append((left_item, right_item, nested))
                else:
                    target.append(left_item > right_item)
        return type(self)(result)

    def __ge__(self, other) -> Tensor:
        if not isinstance(other, type(self)):
            return NotImplemented
        result = []
        stack = [(self.data, other.data, result)]
        while stack:
            left, right, target = stack.pop()
            for left_item, right_item in zip(left, right):
                if isinstance(left_item, list):
                    nested = []
                    target.append(nested)
                    stack.append((left_item, right_item, nested))
                else:
                    target.append(left_item >= right_item)
        return type(self)(result)
