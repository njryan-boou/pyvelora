from __future__ import annotations

import numpy as np

def format_float(value: float) -> str:
    return np.format_float_positional(float(value), trim="-")


def format_complex(value: complex) -> str:
    real = float(np.real(value))
    imag = float(np.imag(value))
    sign = "+" if imag >= 0 else "-"
    return f"{format_float(real)} {sign} {format_float(abs(imag))}j"


def format_array(data) -> str:
    return np.array2string(
        data,
        formatter={
            "float_kind": format_float,
            "complex_kind": format_complex,
        },
    )


class Base:
    __slots__ = ["data"]

    def __init__(self, data: np.ndarray | list | tuple) -> None:
        self.data = np.asarray(data, dtype=np.float64)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def size(self) -> int:
        return self.data.size

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype
