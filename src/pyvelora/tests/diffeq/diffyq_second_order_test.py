"""Tests for pyvelora/diffyq/second_order.py."""
import numpy as np
import pytest
from pyvelora.diffyq.second_order import solve_second_order


def test_solve_second_order_free_fall():
    """y'' = -g (constant accel), y(0)=0, y'(0)=0 → y(t) = -g/2 * t^2."""
    g = 9.81
    sol = solve_second_order(
        accel=lambda t, y, v: -g,
        t0=0, y0=0, v0=0, tf=1.0,
        t_eval=np.linspace(0, 1, 20),
    )
    assert sol.success
    assert np.isclose(sol.y[0, -1], -g / 2, atol=1e-3)


def test_solve_second_order_constant_velocity():
    """y'' = 0, y(0)=0, y'(0)=2 → y(t) = 2t."""
    sol = solve_second_order(
        accel=lambda t, y, v: 0.0,
        t0=0, y0=0, v0=2.0, tf=1.0,
        t_eval=np.array([0.0, 0.5, 1.0]),
    )
    assert sol.success
    assert np.isclose(sol.y[0, -1], 2.0, atol=1e-5)


def test_solve_second_order_returns_success():
    sol = solve_second_order(
        accel=lambda t, y, v: 0.0,
        t0=0, y0=1.0, v0=0.0, tf=1.0,
    )
    assert sol.success


def test_solve_second_order_harmonic_oscillator():
    """y'' = -y (SHM), y(0)=1, y'(0)=0 → y(t) = cos(t)."""
    sol = solve_second_order(
        accel=lambda t, y, v: -y,
        t0=0, y0=1.0, v0=0.0, tf=np.pi,
        t_eval=np.linspace(0, np.pi, 100),
    )
    assert np.isclose(sol.y[0, -1], -1.0, atol=1e-4)
