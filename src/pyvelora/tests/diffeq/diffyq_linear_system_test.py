"""Tests for pyvelora/diffyq/linear_system.py."""
import numpy as np
import pytest
from pyvelora.core import Matrix
from pyvelora.diffyq.linear_system import solve_linear


def test_solve_linear_constant_system():
    """A = [[0,0],[0,0]], x(0) = [1, 2] → x stays constant."""
    A = [[0, 0], [0, 0]]
    sol = solve_linear(A, x0=[1.0, 2.0], t0=0, tf=1.0,
                       t_eval=np.array([0.0, 0.5, 1.0]))
    assert sol.success
    assert np.allclose(sol.y[:, -1], [1.0, 2.0], atol=1e-6)


def test_solve_linear_single_exponential():
    """A = [[-1]], x(0) = [1] → x(t) = e^(-t)."""
    A = [[-1.0]]
    sol = solve_linear(A, x0=[1.0], t0=0, tf=2.0,
                       t_eval=np.linspace(0, 2, 50))
    assert np.isclose(sol.y[0, -1], np.exp(-2), atol=1e-4)


def test_solve_linear_returns_success():
    A = [[0.0]]
    sol = solve_linear(A, x0=[3.0], t0=0, tf=1.0)
    assert sol.success


def test_solve_linear_t_eval_length():
    A = [[0.0, 0.0], [0.0, 0.0]]
    t_eval = np.linspace(0, 1, 7)
    sol = solve_linear(A, x0=[1.0, 0.0], t0=0, tf=1.0, t_eval=t_eval)
    assert len(sol.t) == 7


def test_solve_linear_accepts_matrix_input():
    A = Matrix([[0.0, 0.0], [0.0, 0.0]])
    sol = solve_linear(A, x0=[1.0, 2.0], t0=0, tf=1.0)
    assert sol.success


def test_solve_linear_non_square_raises():
    A = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    with pytest.raises(ValueError):
        solve_linear(A, x0=[1.0, 2.0], t0=0, tf=1.0)
