"""Vector implementation for 1D numeric arrays with vector operations. and coordinate parsing."""
from __future__ import annotations

from typing import TYPE_CHECKING

import math

from pyvelora.core.array_base import Base, VectorData
from pyvelora.core.vector.indexing import VectorIndexing as indexing
from pyvelora.core.vector.arithmetic import VectorArithmetic as arithmetic
from pyvelora.core.vector.comparison import VectorComparison as comparison

if TYPE_CHECKING:
    from pyvelora.core.matrix import Matrix


class Vector(Base, indexing, arithmetic, comparison):
    """One-dimensional numeric array with vector operations."""

    def __init__(self, data, type: str | None = None, degrees: bool = False) -> None:
        values = data
        if type is not None:
            coord_type = type.lower()
            values = list(data)

            if coord_type == "polar":
                if len(values) != 2:
                    raise ValueError("Polar coordinates require [r, theta]")
                radius, theta = values
                if degrees:
                    theta = math.radians(theta)
                values = [radius * math.cos(theta), radius * math.sin(theta)]

            elif coord_type == "spherical":
                if len(values) != 3:
                    raise ValueError("Spherical coordinates require [r, theta, phi]")
                radius, theta, phi = values
                if degrees:
                    theta = math.radians(theta)
                    phi = math.radians(phi)
                values = [
                    radius * math.sin(theta) * math.cos(phi),
                    radius * math.sin(theta) * math.sin(phi),
                    radius * math.cos(theta),
                ]

            elif coord_type == "cylindrical":
                if len(values) != 3:
                    raise ValueError("Cylindrical coordinates require [rho, phi, z]")
                rho, phi, z_value = values
                if degrees:
                    phi = math.radians(phi)
                values = [rho * math.cos(phi), rho * math.sin(phi), z_value]

            elif coord_type == "complex":
                if len(values) != 2:
                    raise ValueError("Complex coordinates require [real, imag]")
                real, imag = values
                values = [complex(real, imag)]

            else:
                raise ValueError(f"Unknown coordinate type: {type}")

        super().__init__(values)
        if self.ndim != 1:
            raise ValueError("Vector must be 1D")
        if any(isinstance(item, complex) for item in self.data):
            self.data = VectorData([complex(item) for item in self.data])
        else:
            self.data = VectorData([float(item) for item in self.data])

    def __str__(self) -> str:
        values = []
        for value in self.data:
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

        if getattr(self, "_display_as_row", False):
            return "⎡ " + "  ".join(values) + " ⎤"
        return "[" + " ".join(values) + "]"

    def __repr__(self) -> str:
        values = []
        for value in self.data:
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
        return "Vector([" + " ".join(values) + "])"
