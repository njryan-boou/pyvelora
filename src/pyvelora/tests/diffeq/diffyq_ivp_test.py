"""Tests for pyvelora/diffyq/ivp.py."""
import numpy as np
import pytest
from pyvelora.diffyq.ivp import solve_ivp_wrapper, ODESolution


def test_solve_ivp_wrapper_exponential_decay():
    """dy/dt = -y, y(0) = 1 → y(t) = e^(-t)"""
    sol = solve_ivp_wrapper(
        f=lambda t, y: [-y[0]],
        t_span=(0, 2),
        y0=1.0,
        t_eval=np.linspace(0, 2, 50),
    )
    assert sol.success
    assert np.isclose(sol.y[0, -1], np.exp(-2), atol=1e-4)


def test_solve_ivp_wrapper_returns_ode_solution():
    sol = solve_ivp_wrapper(
        f=lambda t, y: [0.0],
        t_span=(0, 1),
        y0=5.0,
    )
    assert isinstance(sol, ODESolution)


def test_ode_solution_final_scalar():
    sol = solve_ivp_wrapper(
        f=lambda t, y: [0.0],
        t_span=(0, 1),
        y0=3.0,
        t_eval=np.array([0.0, 0.5, 1.0]),
    )
    assert np.isclose(sol.final, 3.0, atol=1e-5)


def test_ode_solution_final_vector():
    sol = solve_ivp_wrapper(
        f=lambda t, y: [0.0, 0.0],
        t_span=(0, 1),
        y0=[2.0, 4.0],
        t_eval=np.array([0.0, 0.5, 1.0]),
    )
    assert np.allclose(sol.final, [2.0, 4.0], atol=1e-5)


def test_ode_solution_at():
    sol = solve_ivp_wrapper(
        f=lambda t, y: [0.0],
        t_span=(0, 1),
        y0=1.0,
        t_eval=np.array([0.0, 1.0]),
    )
    assert np.isclose(sol.at(0), 1.0, atol=1e-5)


def test_ode_solution_pairs():
    sol = solve_ivp_wrapper(
        f=lambda t, y: [0.0],
        t_span=(0, 1),
        y0=1.0,
        t_eval=np.array([0.0, 0.5, 1.0]),
    )
    pairs = sol.pairs()
    assert len(pairs) == 3
    assert pairs[0][0] == pytest.approx(0.0)
