from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyvelora.core.tensor.tensor import Tensor


class TensorArithmetic:
    """Arithmetic operations mixin for Tensor class."""

    def __neg__(self) -> Tensor:
        result = []
        stack = [(self.data, result)]
        while stack:
            source, target = stack.pop()
            for item in source:
                if isinstance(item, list):
                    nested = []
                    target.append(nested)
                    stack.append((item, nested))
                else:
                    target.append(-item)
        return type(self)(result)

    def __pos__(self) -> Tensor:
        result = []
        stack = [(self.data, result)]
        while stack:
            source, target = stack.pop()
            for item in source:
                if isinstance(item, list):
                    nested = []
                    target.append(nested)
                    stack.append((item, nested))
                else:
                    target.append(+item)
        return type(self)(result)

    def __abs__(self) -> Tensor:
        result = []
        stack = [(self.data, result)]
        while stack:
            source, target = stack.pop()
            for item in source:
                if isinstance(item, list):
                    nested = []
                    target.append(nested)
                    stack.append((item, nested))
                else:
                    target.append(abs(item))
        return type(self)(result)

    def __add__(self, other) -> Tensor:
        if not isinstance(other, type(self)):
            return NotImplemented
        if self.shape != other.shape:
            raise ValueError("Tensor shapes must be the same for addition")
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
                    target.append(left_item + right_item)
        return type(self)(result)

    def __sub__(self, other) -> Tensor:
        if not isinstance(other, type(self)):
            return NotImplemented
        if self.shape != other.shape:
            raise ValueError("Tensor shapes must be the same for subtraction")
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
                    target.append(left_item - right_item)
        return type(self)(result)

    def __mul__(self, other) -> Tensor:
        if isinstance(other, (int, float)):
            result = []
            stack = [(self.data, result)]
            while stack:
                source, target = stack.pop()
                for item in source:
                    if isinstance(item, list):
                        nested = []
                        target.append(nested)
                        stack.append((item, nested))
                    else:
                        target.append(item * other)
            return type(self)(result)
        if isinstance(other, type(self)):
            if self.shape != other.shape:
                raise ValueError("Tensor shapes must be the same for multiplication")
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
                        target.append(left_item * right_item)
            return type(self)(result)
        return NotImplemented

    def __truediv__(self, other) -> Tensor:
        if isinstance(other, (int, float)):
            if other == 0.0:
                raise ZeroDivisionError("Cannot divide by zero")
            result = []
            stack = [(self.data, result)]
            while stack:
                source, target = stack.pop()
                for item in source:
                    if isinstance(item, list):
                        nested = []
                        target.append(nested)
                        stack.append((item, nested))
                    else:
                        target.append(item / other)
            return type(self)(result)
        if isinstance(other, type(self)):
            if self.shape != other.shape:
                raise ValueError("Tensor shapes must be the same for division")
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
                        if right_item == 0:
                            raise ZeroDivisionError("Cannot divide by zero")
                        target.append(left_item / right_item)
            return type(self)(result)
        return NotImplemented

    def __radd__(self, other) -> Tensor:
        if not isinstance(other, type(self)):
            return NotImplemented
        if self.shape != other.shape:
            raise ValueError("Tensor shapes must be the same for addition")
        result = []
        stack = [(other.data, self.data, result)]
        while stack:
            left, right, target = stack.pop()
            for left_item, right_item in zip(left, right):
                if isinstance(left_item, list):
                    nested = []
                    target.append(nested)
                    stack.append((left_item, right_item, nested))
                else:
                    target.append(left_item + right_item)
        return type(self)(result)

    def __rsub__(self, other) -> Tensor:
        if not isinstance(other, type(self)):
            return NotImplemented
        if self.shape != other.shape:
            raise ValueError("Tensor shapes must be the same for subtraction")
        result = []
        stack = [(other.data, self.data, result)]
        while stack:
            left, right, target = stack.pop()
            for left_item, right_item in zip(left, right):
                if isinstance(left_item, list):
                    nested = []
                    target.append(nested)
                    stack.append((left_item, right_item, nested))
                else:
                    target.append(left_item - right_item)
        return type(self)(result)

    def __rmul__(self, other) -> Tensor:
        if isinstance(other, (int, float)):
            result = []
            stack = [(self.data, result)]
            while stack:
                source, target = stack.pop()
                for item in source:
                    if isinstance(item, list):
                        nested = []
                        target.append(nested)
                        stack.append((item, nested))
                    else:
                        target.append(other * item)
            return type(self)(result)
        if isinstance(other, type(self)):
            if self.shape != other.shape:
                raise ValueError("Tensor shapes must be the same for multiplication")
            result = []
            stack = [(other.data, self.data, result)]
            while stack:
                left, right, target = stack.pop()
                for left_item, right_item in zip(left, right):
                    if isinstance(left_item, list):
                        nested = []
                        target.append(nested)
                        stack.append((left_item, right_item, nested))
                    else:
                        target.append(left_item * right_item)
            return type(self)(result)
        return NotImplemented

    def __rtruediv__(self, other) -> Tensor:
        if isinstance(other, (int, float)):
            result = []
            stack = [(self.data, result)]
            while stack:
                source, target = stack.pop()
                for item in source:
                    if isinstance(item, list):
                        nested = []
                        target.append(nested)
                        stack.append((item, nested))
                    else:
                        if item == 0:
                            raise ZeroDivisionError("Cannot divide by zero")
                        target.append(other / item)
            return type(self)(result)
        if isinstance(other, type(self)):
            if self.shape != other.shape:
                raise ValueError("Tensor shapes must be the same for division")
            result = []
            stack = [(other.data, self.data, result)]
            while stack:
                left, right, target = stack.pop()
                for left_item, right_item in zip(left, right):
                    if isinstance(left_item, list):
                        nested = []
                        target.append(nested)
                        stack.append((left_item, right_item, nested))
                    else:
                        if right_item == 0:
                            raise ZeroDivisionError("Cannot divide by zero")
                        target.append(left_item / right_item)
            return type(self)(result)
        return NotImplemented

    def __iadd__(self, other) -> Tensor:
        if not isinstance(other, type(self)):
            return NotImplemented
        if self.shape != other.shape:
            raise ValueError("Tensor shapes must be the same for addition")
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
                    target.append(left_item + right_item)
        self.data = result
        return self

    def __isub__(self, other) -> Tensor:
        if not isinstance(other, type(self)):
            return NotImplemented
        if self.shape != other.shape:
            raise ValueError("Tensor shapes must be the same for subtraction")
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
                    target.append(left_item - right_item)
        self.data = result
        return self

    def __imul__(self, other) -> Tensor:
        if isinstance(other, (int, float)):
            result = []
            stack = [(self.data, result)]
            while stack:
                source, target = stack.pop()
                for item in source:
                    if isinstance(item, list):
                        nested = []
                        target.append(nested)
                        stack.append((item, nested))
                    else:
                        target.append(item * other)
            self.data = result
            return self
        if isinstance(other, type(self)):
            if self.shape != other.shape:
                raise ValueError("Tensor shapes must be the same for multiplication")
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
                        target.append(left_item * right_item)
            self.data = result
            return self
        return NotImplemented

    def __itruediv__(self, other) -> Tensor:
        if isinstance(other, (int, float)):
            if other == 0.0:
                raise ZeroDivisionError("Cannot divide by zero")
            result = []
            stack = [(self.data, result)]
            while stack:
                source, target = stack.pop()
                for item in source:
                    if isinstance(item, list):
                        nested = []
                        target.append(nested)
                        stack.append((item, nested))
                    else:
                        target.append(item / other)
            self.data = result
            return self
        if isinstance(other, type(self)):
            if self.shape != other.shape:
                raise ValueError("Tensor shapes must be the same for division")
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
                        if right_item == 0:
                            raise ZeroDivisionError("Cannot divide by zero")
                        target.append(left_item / right_item)
            self.data = result
            return self
        return NotImplemented
