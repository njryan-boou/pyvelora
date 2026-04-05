"""Tests for pyvelora/diffyq/system.py."""
import numpy as np
import pytest
from pyvelora.diffyq.system import solve_system


def test_solve_system_constant_system():
    """dx/dt = 0, x(0) = [1, 2] → x stays constant."""
    sol = solve_system(
        F=lambda t, x: [0.0, 0.0],
        t0=0,
        x0=[1.0, 2.0],
        tf=1.0,
        t_eval=np.array([0.0, 0.5, 1.0]),
    )
    assert sol.success
    assert np.allclose(sol.y[:, -1], [1.0, 2.0], atol=1e-6)


def test_solve_system_exponential_growth():
    """dx/dt = x, x(0) = 1 → x(t) = e^t."""
    sol = solve_system(
        F=lambda t, x: [x[0]],
        t0=0,
        x0=[1.0],
        tf=1.0,
        t_eval=np.linspace(0, 1, 20),
    )
    assert np.isclose(sol.y[0, -1], np.e, atol=1e-4)


def test_solve_system_returns_success_flag():
    sol = solve_system(
        F=lambda t, x: [0.0],
        t0=0,
        x0=1.0,
        tf=1.0,
    )
    assert sol.success


def test_solve_system_t_eval_respected():
    t_eval = np.linspace(0, 1, 11)
    sol = solve_system(
        F=lambda t, x: [0.0],
        t0=0,
        x0=1.0,
        tf=1.0,
        t_eval=t_eval,
    )
    assert len(sol.t) == 11
