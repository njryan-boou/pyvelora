from __future__ import annotations

from collections.abc import Sequence

from pyvelora.core.array_base import MatrixData, TensorData, VectorData


def linspace(start: float, stop: float, num: int = 50) -> list[float]:
    """Return evenly spaced numbers over a specified interval."""
    if num <= 0:
        raise ValueError("num must be positive")
    if num == 1:
        return [float(start)]
    step = (stop - start) / (num - 1)
    return [float(start + i * step) for i in range(num)]

def logspace(start: float, stop: float, num: int = 50) -> list[float]:
    """Return values evenly spaced in log10-space between powers start and stop."""
    return [10.0 ** x for x in linspace(start, stop, num)]
   
def arange(start: float, stop: float, step: float) -> list[float]:
    """Return evenly spaced values within a given interval."""
    if step == 0:
        raise ValueError("step must be non-zero")
    values = []
    current = float(start)
    if step > 0:
        while current < stop:
            values.append(current)
            current += step
    else:
        while current > stop:
            values.append(current)
            current += step
    return values

def meshgrid(x: list[float], y: list[float]) -> tuple[list[list[float]], list[list[float]]]:
    """Return coordinate matrices with ij-indexing."""
    x_values = list(x)
    y_values = list(y)
    gx = MatrixData([[x_values[i] for _ in y_values] for i in range(len(x_values))])
    gy = MatrixData([[y_values[j] for j in range(len(y_values))] for _ in x_values])
    return gx, gy


def _as_axis_list(value, ndim: int) -> list[float]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        items = list(value)
        if len(items) != ndim:
            raise ValueError("Sequence inputs must match ndim")
        return [float(v) for v in items]
    return [float(value) for _ in range(ndim)]


def _wrap_nested(data):
    depth = 0
    current = data
    while isinstance(current, list):
        depth += 1
        if not current:
            break
        current = current[0]
    if depth <= 1:
        return VectorData(data)
    if depth == 2:
        return MatrixData(data)
    return TensorData(data)


def _build_nd_grids(axes: list[list[float]]) -> list:
    ndim = len(axes)
    shape = [len(axis) for axis in axes]

    def build_grid(grid_axis: int, depth: int, index_prefix: tuple[int, ...]):
        if depth == ndim:
            return axes[grid_axis][index_prefix[grid_axis]]
        return [
            build_grid(grid_axis, depth + 1, index_prefix + (i,))
            for i in range(shape[depth])
        ]

    return [_wrap_nested(build_grid(axis_idx, 0, ())) for axis_idx in range(ndim)]


def linspace_nd(start: float | Sequence[float], stop: float | Sequence[float], num: int = 50, ndim: int = 1):
    """Return N-dimensional linspace grids using ij-indexing."""
    if ndim < 1:
        raise ValueError("ndim must be >= 1")
    starts = _as_axis_list(start, ndim)
    stops = _as_axis_list(stop, ndim)
    axes = [linspace(starts[i], stops[i], num) for i in range(ndim)]
    return _build_nd_grids(axes)


def logspace_nd(start: float | Sequence[float], stop: float | Sequence[float], num: int = 50, ndim: int = 1):
    """Return N-dimensional logspace grids using ij-indexing."""
    if ndim < 1:
        raise ValueError("ndim must be >= 1")
    starts = _as_axis_list(start, ndim)
    stops = _as_axis_list(stop, ndim)
    axes = [logspace(starts[i], stops[i], num) for i in range(ndim)]
    return _build_nd_grids(axes)


def arange_nd(start: float | Sequence[float], stop: float | Sequence[float], step: float | Sequence[float], ndim: int = 1):
    """Return N-dimensional arange grids using ij-indexing."""
    if ndim < 1:
        raise ValueError("ndim must be >= 1")
    starts = _as_axis_list(start, ndim)
    stops = _as_axis_list(stop, ndim)
    steps = _as_axis_list(step, ndim)
    axes = [arange(starts[i], stops[i], steps[i]) for i in range(ndim)]
    return _build_nd_grids(axes)




