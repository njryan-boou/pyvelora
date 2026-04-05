"""Matrix implementation for 2D numeric arrays with linear algebra operations."""
from __future__ import annotations
from typing import TYPE_CHECKING

from pyvelora.core.array_base import Base
from pyvelora.core.matrix.indexing import MatrixIndexing
from pyvelora.core.matrix.arithmetic import MatrixArithmetic
from pyvelora.core.matrix.comparison import MatrixComparison

if TYPE_CHECKING:
    from pyvelora.core.vector.vector import Vector


class Matrix(Base, MatrixIndexing, MatrixArithmetic, MatrixComparison):
    """Two-dimensional numeric array with linear algebra operations."""

    def __init__(self, data: list[list[float]] | list[list[complex]]) -> None:
        super().__init__(data)
        if self.ndim != 2:
            raise ValueError("Matrix must be 2D")
        
    def __str__(self) -> str:
        rows = []
        for index in range(self.shape[0]):
            parts = []
            for value in self.data[index]:
                if isinstance(value, complex):
                    real = value.real
                    imag = value.imag
                    if imag == 0:
                        if float(real).is_integer():
                            parts.append(str(int(real)))
                        else:
                            parts.append(f"{float(real):.10f}".rstrip("0").rstrip("."))
                    elif real == 0:
                        magnitude = abs(imag)
                        if float(magnitude).is_integer():
                            magnitude_text = str(int(magnitude))
                        else:
                            magnitude_text = f"{float(magnitude):.10f}".rstrip("0").rstrip(".")
                        parts.append(("-" if imag < 0 else "") + magnitude_text + "j")
                    else:
                        if float(real).is_integer():
                            real_text = str(int(real))
                        else:
                            real_text = f"{float(real):.10f}".rstrip("0").rstrip(".")
                        magnitude = abs(imag)
                        if float(magnitude).is_integer():
                            imag_text = str(int(magnitude))
                        else:
                            imag_text = f"{float(magnitude):.10f}".rstrip("0").rstrip(".")
                        parts.append(f"{real_text} {'+' if imag > 0 else '-'} {imag_text}j")
                elif isinstance(value, (int, float)):
                    if float(value).is_integer():
                        parts.append(str(int(value)))
                    else:
                        parts.append(f"{float(value):.10f}".rstrip("0").rstrip("."))
                else:
                    parts.append(str(value))

            if self.shape[0] == 1 or index == 0:
                rows.append("⎡ " + "  ".join(parts) + " ⎤")
            elif index == self.shape[0] - 1:
                rows.append("⎣ " + "  ".join(parts) + " ⎦")
            else:
                rows.append("⎢ " + "  ".join(parts) + " ⎥")

        if not rows:
            return "[]"
        if len(rows) == 1:
            return rows[0]
        rows[0] = rows[0].replace("⎢", "⎡").replace("⎥", "⎤")
        rows[-1] = rows[-1].replace("⎢", "⎣").replace("⎥", "⎦")
        return "\n".join(rows)
    
    def __repr__(self) -> str:
        rows = []
        for row in self.data:
            parts = []
            for value in row:
                if isinstance(value, complex):
                    real = value.real
                    imag = value.imag
                    if imag == 0:
                        if float(real).is_integer():
                            parts.append(str(int(real)))
                        else:
                            parts.append(f"{float(real):.10f}".rstrip("0").rstrip("."))
                    elif real == 0:
                        magnitude = abs(imag)
                        if float(magnitude).is_integer():
                            magnitude_text = str(int(magnitude))
                        else:
                            magnitude_text = f"{float(magnitude):.10f}".rstrip("0").rstrip(".")
                        parts.append(("-" if imag < 0 else "") + magnitude_text + "j")
                    else:
                        if float(real).is_integer():
                            real_text = str(int(real))
                        else:
                            real_text = f"{float(real):.10f}".rstrip("0").rstrip(".")
                        magnitude = abs(imag)
                        if float(magnitude).is_integer():
                            imag_text = str(int(magnitude))
                        else:
                            imag_text = f"{float(magnitude):.10f}".rstrip("0").rstrip(".")
                        parts.append(f"{real_text} {'+' if imag > 0 else '-'} {imag_text}j")
                elif isinstance(value, (int, float)):
                    if float(value).is_integer():
                        parts.append(str(int(value)))
                    else:
                        parts.append(f"{float(value):.10f}".rstrip("0").rstrip("."))
                else:
                    parts.append(str(value))
            rows.append("[" + " ".join(parts) + "]")
        return "Matrix([" + "\n ".join(rows) + "])"

    def format_row(self, index: int) -> str:
        """Return a single matrix row using bracketed display formatting."""
        if not 0 <= index < self.shape[0]:
            raise IndexError("Row index out of bounds")
        values = []
        for value in self.data[index]:
            if isinstance(value, complex):
                real = value.real
                imag = value.imag
                if imag == 0:
                    if float(real).is_integer():
                        values.append(str(int(real)))
                    else:
                        values.append(f"{float(real):.10f}".rstrip("0").rstrip("."))
                elif real == 0:
                    magnitude = abs(imag)
                    if float(magnitude).is_integer():
                        magnitude_text = str(int(magnitude))
                    else:
                        magnitude_text = f"{float(magnitude):.10f}".rstrip("0").rstrip(".")
                    values.append(("-" if imag < 0 else "") + magnitude_text + "j")
                else:
                    if float(real).is_integer():
                        real_text = str(int(real))
                    else:
                        real_text = f"{float(real):.10f}".rstrip("0").rstrip(".")
                    magnitude = abs(imag)
                    if float(magnitude).is_integer():
                        imag_text = str(int(magnitude))
                    else:
                        imag_text = f"{float(magnitude):.10f}".rstrip("0").rstrip(".")
                    values.append(f"{real_text} {'+' if imag > 0 else '-'} {imag_text}j")
            elif isinstance(value, (int, float)):
                if float(value).is_integer():
                    values.append(str(int(value)))
                else:
                    values.append(f"{float(value):.10f}".rstrip("0").rstrip("."))
            else:
                values.append(str(value))
        if self.shape[0] == 1 or index == 0:
            return "⎡ " + "  ".join(values) + " ⎤"
        if index == self.shape[0] - 1:
            return "⎣ " + "  ".join(values) + " ⎦"
        return "⎢ " + "  ".join(values) + " ⎥"

    def get_row(self, index: int) -> Vector:
        """Return the specified row as a Vector displayed as a row vector."""
        from pyvelora.core.vector.vector import Vector
        if not 0 <= index < self.shape[0]:
            raise IndexError("Row index out of bounds")
        row_vector = Vector(self.data[index])
        row_vector._display_as_row = True
        return row_vector

    def get_column(self, index: int) -> Vector:
        """Return the specified column as a Vector."""
        from pyvelora.core.vector.vector import Vector
        if not 0 <= index < self.shape[1]:
            raise IndexError("Column index out of bounds")
        return Vector([row[index] for row in self.data])
