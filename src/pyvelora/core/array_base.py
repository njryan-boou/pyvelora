from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy


class VectorData(list):
    @property
    def real(self):
        return [value.real if isinstance(value, complex) else value for value in self]

    @property
    def shape(self):
        return (len(self),)

    def flatten(self):
        return VectorData(self[:])

    def reshape(self, *shape):
        flat = list(self)
        total = 1
        for d in shape:
            total *= d
        if total != len(flat):
            raise ValueError(f"Cannot reshape VectorData of size {len(flat)} into shape {shape}")
        idx = [0]
        def _build(s):
            if len(s) == 1:
                result = []
                for _ in range(s[0]):
                    result.append(flat[idx[0]])
                    idx[0] += 1
                return result
            return [_build(s[1:]) for _ in range(s[0])]
        nested = _build(list(shape))
        if len(shape) == 1:
            return VectorData(nested)
        elif len(shape) == 2:
            return MatrixData(nested)
        else:
            return TensorData(nested)

    def astype(self, _dtype):
        return VectorData(self[:])


class MatrixData(list):
    def __getitem__(self, key):
        if isinstance(key, tuple):
            row_key, col_key = key
            rows = list.__getitem__(self, row_key) if isinstance(row_key, slice) else [list.__getitem__(self, row_key)]
            if isinstance(col_key, slice):
                values = [[row[index] for index in range(*col_key.indices(len(row)))] for row in rows]
                return MatrixData(values) if isinstance(row_key, slice) else VectorData(values[0])
            values = [row[col_key] for row in rows]
            return VectorData(values) if isinstance(row_key, slice) else values[0]

        value = list.__getitem__(self, key)
        if isinstance(key, slice):
            return MatrixData([list(row) for row in value])
        return value

    @property
    def T(self):
        if not self:
            return MatrixData([])
        return MatrixData([
            [self[row][col] for row in range(len(self))]
            for col in range(len(self[0]))
        ])

    def flatten(self):
        flattened = []
        for row in self:
            for value in row:
                flattened.append(value)
        return VectorData(flattened)

    @property
    def shape(self):
        if not self:
            return (0, 0)
        return (len(self), len(self[0]))

    def astype(self, _dtype):
        return MatrixData([row[:] for row in self])

    def tolist(self):
        return [list(row) for row in self]


def _apply_elementwise(data, other, op):
    """Recursively apply op(a, b) elementwise across nested list structures."""
    if isinstance(other, (int, float, complex)):
        if isinstance(data, list):
            result = type(data)()
            for item in data:
                result.append(_apply_elementwise(item, other, op))
            return result
        return op(data, other)
    if isinstance(data, list) and isinstance(other, list):
        result = type(data)()
        for a, b in zip(data, other):
            result.append(_apply_elementwise(a, b, op))
        return result
    return op(data, other)


class TensorData(list):
    @property
    def shape(self):
        shape = []
        current = self
        while isinstance(current, list):
            shape.append(len(current))
            if not current:
                break
            current = current[0]
        return tuple(shape)

    def __mul__(self, other):
        return _apply_elementwise(self, other, lambda a, b: a * b)

    def __rmul__(self, other):
        return _apply_elementwise(self, other, lambda a, b: b * a)

    def __sub__(self, other):
        return _apply_elementwise(self, other, lambda a, b: a - b)

    def __rsub__(self, other):
        return _apply_elementwise(self, other, lambda a, b: b - a)

    def __truediv__(self, other):
        return _apply_elementwise(self, other, lambda a, b: a / b)

    def __rtruediv__(self, other):
        return _apply_elementwise(self, other, lambda a, b: b / a)

    def flatten(self):
        flattened = []
        def _collect(data):
            for item in data:
                if isinstance(item, list):
                    _collect(item)
                else:
                    flattened.append(item)
        _collect(self)
        return VectorData(flattened)

    def astype(self, _dtype):
        source = deepcopy(list(self))
        depth = 0
        probe = source
        while isinstance(probe, list):
            depth += 1
            if not probe:
                break
            probe = probe[0]
        if depth <= 1:
            return VectorData(source)
        if depth == 2:
            return MatrixData([list(row) for row in source])
        wrapped = []
        stack = [(source, wrapped)]
        while stack:
            current_source, current_target = stack.pop()
            for item in current_source:
                if isinstance(item, list):
                    item_depth = 0
                    item_probe = item
                    while isinstance(item_probe, list):
                        item_depth += 1
                        if not item_probe:
                            break
                        item_probe = item_probe[0]
                    if item_depth <= 1:
                        current_target.append(VectorData(item))
                    elif item_depth == 2:
                        current_target.append(MatrixData([list(row) for row in item]))
                    else:
                        nested = TensorData()
                        current_target.append(nested)
                        stack.append((item, nested))
                else:
                    current_target.append(item)
        return TensorData(wrapped)


def format_float(value: float) -> str:
    """Format a float, removing trailing zeros."""
    if float(value).is_integer():
        return str(int(value))
    return f"{float(value):.10f}".rstrip("0").rstrip(".")


def format_complex(value: complex) -> str:
    """Format a complex number with explicit sign and j notation."""
    real = value.real
    imag = value.imag
    if imag == 0:
        return format_float(real)
    if real == 0:
        sign = "-" if imag < 0 else ""
        magnitude = format_float(abs(imag))
        return f"{sign}{magnitude}j"
    sign = "+" if imag > 0 else "-"
    return f"{format_float(real)} {sign} {format_float(abs(imag))}j"


def format_array(data) -> str:
    """Format an array for display using pyvelora's compact numeric style."""
    if isinstance(data, complex):
        return format_complex(data)
    if isinstance(data, (int, float)):
        return format_float(float(data))
    if not isinstance(data, list):
        # Convert other iterables (e.g., numpy arrays) to a plain list
        try:
            data = list(data)
        except TypeError:
            return str(data)
    if not data or not isinstance(data[0], list):
        parts = []
        for value in data:
            if isinstance(value, complex):
                parts.append(format_complex(value))
            elif isinstance(value, (int, float)):
                parts.append(format_float(float(value)))
            else:
                parts.append(str(value))
        return "[" + " ".join(parts) + "]"
    rows = [format_array(row) for row in data]
    return "[" + "\n ".join(rows) + "]"


class Base:
    """Base class for array types in pyvelora, providing common initialization and properties."""
    __slots__ = ["data"]

    def __init__(self, data: list) -> None:
        def to_plain(value):
            if hasattr(value, "data"):
                value = value.data
            if hasattr(value, "tolist"):
                value = value.tolist()
            elif not isinstance(value, list) and isinstance(value, Iterable) and not isinstance(value, (str, bytes, dict)):
                value = list(value)
            if isinstance(value, list):
                return [to_plain(item) for item in value]
            return value

        normalized = to_plain(data)
        if not isinstance(normalized, list):
            self.data = normalized
        else:
            depth = 0
            current = normalized
            while isinstance(current, list):
                depth += 1
                if not current:
                    break
                current = current[0]
            if depth <= 1:
                self.data = VectorData(normalized)
            elif depth == 2:
                self.data = MatrixData([list(row) for row in normalized])
            else:
                wrapped = []
                stack = [(normalized, wrapped)]
                while stack:
                    source, target = stack.pop()
                    for item in source:
                        if isinstance(item, list):
                            item_depth = 0
                            probe = item
                            while isinstance(probe, list):
                                item_depth += 1
                                if not probe:
                                    break
                                probe = probe[0]
                            if item_depth <= 1:
                                target.append(VectorData(item))
                            elif item_depth == 2:
                                target.append(MatrixData([list(row) for row in item]))
                            else:
                                nested = TensorData()
                                target.append(nested)
                                stack.append((item, nested))
                        else:
                            target.append(item)
                self.data = TensorData(wrapped)

    @property
    def shape(self) -> tuple[int, ...]:
        shape = []
        current = self.data
        while isinstance(current, list):
            shape.append(len(current))
            if not current:
                break
            current = current[0]
        return tuple(shape)

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def size(self) -> int:
        count = 0
        stack = [self.data]
        while stack:
            current = stack.pop()
            if isinstance(current, list):
                for item in current:
                    stack.append(item)
            else:
                count += 1
        return count

    @property
    def dtype(self):
        has_complex = False
        has_bool = False
        stack = [self.data]
        while stack:
            current = stack.pop()
            if isinstance(current, list):
                for item in current:
                    stack.append(item)
            elif isinstance(current, complex):
                has_complex = True
                break
            elif isinstance(current, bool):
                has_bool = True
        if has_complex:
            return "complex"
        if has_bool:
            return "bool"
        return "float"

    def __copy__(self) -> Base:
        return type(self)(deepcopy(self.data))

    def __deepcopy__(self, memo) -> Base:
        return type(self)(deepcopy(self.data, memo))

    def copy(self) -> Base:
        return type(self)(deepcopy(self.data))

    def deepcopy(self) -> Base:
        return type(self)(deepcopy(self.data))

    def __bool__(self) -> bool:
        stack = [self.data]
        while stack:
            current = stack.pop()
            if isinstance(current, list):
                for item in current:
                    stack.append(item)
            elif bool(current):
                return True
        return False

    def __float__(self):
        if self.size != 1:
            raise ValueError(f"Cannot convert {type(self).__name__} with shape {self.shape} to float")
        current = self.data
        while isinstance(current, list):
            current = current[0]
        return float(current)

    def __complex__(self):
        if self.size != 1:
            raise ValueError(f"Cannot convert {type(self).__name__} with shape {self.shape} to complex")
        current = self.data
        while isinstance(current, list):
            current = current[0]
        return complex(current)

    def __format__(self, format_spec):
        def format_value(value):
            if isinstance(value, list):
                if not value or not isinstance(value[0], list):
                    parts = []
                    for item in value:
                        parts.append(format_value(item))
                    return "[" + " ".join(parts) + "]"
                rows = []
                for row in value:
                    rows.append(format_value(row))
                return "[" + "\n " + "\n ".join(rows) + "]"
            if isinstance(value, complex):
                real = value.real
                imag = value.imag
                if imag == 0:
                    if float(real).is_integer():
                        return str(int(real))
                    return f"{float(real):.10f}".rstrip("0").rstrip(".")
                if real == 0:
                    magnitude = abs(imag)
                    if float(magnitude).is_integer():
                        magnitude_text = str(int(magnitude))
                    else:
                        magnitude_text = f"{float(magnitude):.10f}".rstrip("0").rstrip(".")
                    return ("-" if imag < 0 else "") + magnitude_text + "j"
                if float(real).is_integer():
                    real_text = str(int(real))
                else:
                    real_text = f"{float(real):.10f}".rstrip("0").rstrip(".")
                magnitude = abs(imag)
                if float(magnitude).is_integer():
                    imag_text = str(int(magnitude))
                else:
                    imag_text = f"{float(magnitude):.10f}".rstrip("0").rstrip(".")
                return f"{real_text} {'+' if imag > 0 else '-'} {imag_text}j"
            if isinstance(value, (int, float)):
                if float(value).is_integer():
                    return str(int(value))
                return f"{float(value):.10f}".rstrip("0").rstrip(".")
            return str(value)

        return f"{type(self).__name__}({format_value(self.data)})"

    def all(self) -> bool:
        stack = [self.data]
        while stack:
            current = stack.pop()
            if isinstance(current, list):
                for item in current:
                    stack.append(item)
            elif not bool(current):
                return False
        return True

    def any(self) -> bool:
        stack = [self.data]
        while stack:
            current = stack.pop()
            if isinstance(current, list):
                for item in current:
                    stack.append(item)
            elif bool(current):
                return True
        return False

    def reshape(self, *shape) -> Base:
        flat_data = []
        stack = [self.data]
        while stack:
            current = stack.pop()
            if isinstance(current, list):
                for item in reversed(current):
                    stack.append(item)
            else:
                flat_data.append(current)
        total = 1
        for dim in shape:
            total *= dim
        if total != len(flat_data):
            raise ValueError("Cannot reshape array to requested shape")
        values = flat_data[:]

        def build(target_shape):
            if not target_shape:
                return values.pop(0)
            return [build(target_shape[1:]) for _ in range(target_shape[0])]

        return type(self)(build(shape))

    def astype(self, dtype) -> Base:
        return type(self)(deepcopy(self.data))

    def flatten(self) -> Base:
        flat_data = []
        stack = [self.data]
        while stack:
            current = stack.pop()
            if isinstance(current, list):
                for item in reversed(current):
                    stack.append(item)
            else:
                flat_data.append(current)
        return type(self)(flat_data)
