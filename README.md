# pyvelora

pyvelora is a lightweight NumPy-based math backend for scientific and simulation workflows.
It provides structured `Vector`, `Matrix`, and `Tensor` classes, linear algebra helpers,
and an ODE module for differential equation workflows.

## What You Get

- Clear, typed array wrappers for vectors, matrices, and tensors
- Loop-based arithmetic operators with predictable behavior
- Linear algebra helpers for decomposition and matrix operations
- ODE solving helpers with first-order and second-order system support

## Philosophy

- Thin abstraction over NumPy
- Predictable data containers with explicit operations
- Reusable core infrastructure for physics/simulation libraries

## Installation

```bash
pip install pyvelora
```

Optional dependencies:

- SciPy for ODE solving
- Matplotlib for ODE plotting

Install them with:

```bash
pip install scipy matplotlib
```

## Core Quick Start

```python
from pyvelora import Vector, Matrix, Tensor

v = Vector([3, 4])
print(v.magnitude())  # 5.0

m = Matrix([[1, 2], [3, 4]])
print(m.transpose())

t = Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print(t.shape)  # (2, 2, 2)
```

## Coordinate Vector Input

`Vector` supports polar, spherical, and cylindrical coordinate input.

```python
from pyvelora import Vector

# Polar input: [r, theta]
v_polar = Vector([2, 45], type="polar", degrees=True)

# Spherical input: [r, theta, phi]
v_spherical = Vector([1, 90, 30], type="spherical", degrees=True)

# Cylindrical input: [rho, phi, z]
v_cyl = Vector([2, 90, 3], type="cylindrical", degrees=True)
```

Angles are interpreted as radians unless `degrees=True` is passed.

## Linear Algebra Utilities

```python
from pyvelora import Matrix
from pyvelora.linalg import determinant, inverse, solve
from pyvelora.core import Vector

A = Matrix([[2, 1], [5, 3]])
b = Vector([4, 11])

print(determinant(A))
print(inverse(A))
print(solve(A, b))
```

## ODE Module

The ODE API is available under `pyvelora.diffeq.ode`.

```python
import numpy as np
from pyvelora.diffeq.ode import solve, second_order

# First-order system: y' = -y
sol = solve(lambda t, y: -y, (0, 2), [1.0], t_eval=np.linspace(0, 2, 11))
print(sol.final())

# Convert second-order equation y'' = -y to first-order system
sho = second_order(lambda t, y, v: -y)
sho_sol = solve(sho, (0, 2 * np.pi), [1.0, 0.0])
print(sho_sol.final())
```

## Error Types

The package provides reusable exception types:

- `PyveloraError`
- `ShapeError`
- `DimensionError`

```python
from pyvelora import Vector, ShapeError

try:
    _ = Vector([1, 2, 3]) + Vector([1, 2])
except ShapeError as exc:
    print(exc)
```

## Development

Run the test suite:

```bash
python -m pytest src/pyvelora/tests -q
```

Build source and wheel distributions:

```bash
python -m build --no-isolation
```

## Project Status

pyvelora is currently alpha-stage and actively evolving.
Core APIs are usable, but minor interface changes may still happen between releases.