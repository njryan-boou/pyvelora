"""Tests for the ODE module (differential equation solving)."""

import pytest
import numpy as np
from pyvelora.diffeq.ode import solve, Solution, second_order, linear
from pyvelora.core import Vector, Matrix


class TestSolution:
    """Test the Solution class."""
    
    def test_solution_creation(self):
        """Test creating a Solution object."""
        t = np.array([0, 1, 2, 3])
        y = np.array([[1, 5], [2, 6], [3, 7], [4, 8]])
        sol = Solution(t, y)
        
        assert np.allclose(sol.t, t)
        assert np.allclose(sol.y, y)
    
    def test_solution_final(self):
        """Test getting the final state of a solution."""
        t = np.array([0, 1, 2, 3])
        y = np.array([[1, 5], [2, 6], [3, 7], [4, 8]])
        sol = Solution(t, y)
        
        final = sol.final()
        assert np.allclose(final, np.array([4, 8]))
    
    def test_solution_with_scalar_time(self):
        """Test Solution with scalar values."""
        t = [0, 0.5, 1.0]
        y = [[1.0, 2.0, 3.0]]
        sol = Solution(t, y)
        
        assert len(sol.t) == 3
        assert len(sol.y) == 1


class TestSecondOrderConversion:
    """Test the second_order decorator for converting 2nd order to 1st order ODEs."""
    
    def test_simple_harmonic_oscillator(self):
        """Test second order conversion with harmonic oscillator: y'' = -y."""
        def f(t, y, v):
            return -y
        
        system = second_order(f)
        
        t = 0
        Y = [1.0, 0.0]  # Initial position 1, velocity 0
        dY = system(t, Y)
        
        assert dY[0] == 0.0  # v' = v = 0
        assert dY[1] == -1.0  # y' = -y = -1
    
    def test_damped_oscillator(self):
        """Test second order conversion with damped oscillator: y'' + 0.5*y' + y = 0."""
        def f(t, y, v):
            return -0.5 * v - y
        
        system = second_order(f)
        
        t = 0
        Y = [1.0, 0.0]
        dY = system(t, Y)
        
        assert dY[0] == 0.0  # v
        assert dY[1] == -1.0  # y'' = -0.5*0 - 1 = -1
    
    def test_forced_oscillator(self):
        """Test second order conversion with forced oscillator: y'' + y = sin(t)."""
        def f(t, y, v):
            return np.sin(t) - y
        
        system = second_order(f)
        
        t = 0
        Y = [1.0, 0.5]
        dY = system(t, Y)
        
        assert dY[0] == 0.5  # v' = v
        assert np.isclose(dY[1], np.sin(t) - Y[0])  # y'' = sin(t) - y


class TestLinearSystemCreation:
    """Test the linear system creator for linear ODE systems."""
    
    def test_linear_system_creation_2x2(self):
        """Test creating a linear system with 2x2 matrix."""
        A = [[1, 2], [3, 4]]
        system = linear(A)
        
        # Test the system: x' = Ax
        t = 0
        x = Vector([1.0, 1.0])
        dx = system(t, x)
        
        expected = Vector([1*1 + 2*1, 3*1 + 4*1])
        assert np.allclose(dx.data, expected.data)
    
    def test_linear_system_identity(self):
        """Test linear system with identity matrix: x' = x."""
        A = [[1, 0], [0, 1]]
        system = linear(A)
        
        t = 0
        x = Vector([2.0, 3.0])
        dx = system(t, x)
        
        assert np.allclose(dx.data, x.data)
    
    def test_linear_system_diagonal(self):
        """Test linear system with diagonal matrix."""
        A = [[1, 0, 0], [0, 2, 0], [0, 0, 3]]
        system = linear(A)
        
        t = 0
        x = Vector([1.0, 1.0, 1.0])
        dx = system(t, x)
        
        assert np.allclose(dx.data, [1.0, 2.0, 3.0])
    
    def test_linear_system_zero_matrix(self):
        """Test linear system with zero matrix: x' = 0."""
        A = [[0, 0], [0, 0]]
        system = linear(A)
        
        t = 0
        x = Vector([1.0, 2.0])
        dx = system(t, x)
        
        assert np.allclose(dx.data, [0.0, 0.0])
    
    def test_linear_system_returns_vector(self):
        """Test that linear system returns a Vector."""
        A = [[1, 0], [0, 1]]
        system = linear(A)
        
        t = 0
        x = Vector([1.0, 2.0])
        dx = system(t, x)
        
        assert isinstance(dx, Vector)


class TestSolveFunction:
    """Test the solve function for solving ODEs."""
    
    def test_exponential_decay(self):
        """Test solving exponential decay: y' = -y."""
        def f(t, y):
            return -y
        
        t_span = (0, 1)
        y0 = [1.0]
        
        sol = solve(f, t_span, y0, dense_output=True)
        
        assert isinstance(sol, Solution)
        assert len(sol.t) > 1
        assert len(sol.y) > 1
        # Initial condition should match
        assert np.isclose(sol.y[0][0], 1.0, atol=1e-6)
        # Final value should be < initial (decay)
        assert sol.y[-1][0] < 1.0
        # Should be approximately e^(-1) ≈ 0.368
        assert np.isclose(sol.y[-1][0], np.exp(-1), atol=0.01)
    
    def test_exponential_growth(self):
        """Test solving exponential growth: y' = y."""
        def f(t, y):
            return y
        
        t_span = (0, 1)
        y0 = [1.0]
        
        sol = solve(f, t_span, y0, dense_output=True)
        
        assert isinstance(sol, Solution)
        # Final value should be > initial (growth)
        assert sol.y[-1][0] > 1.0
        # Should be approximately e^1 ≈ 2.718
        assert np.isclose(sol.y[-1][0], np.exp(1), atol=0.01)
    
    def test_linspace_with_t_eval(self):
        """Test solve with explicit time evaluation points."""
        def f(t, y):
            return -y
        
        t_span = (0, 1)
        y0 = [1.0]
        t_eval = np.linspace(0, 1, 11)
        
        sol = solve(f, t_span, y0, t_eval=t_eval)
        
        assert isinstance(sol, Solution)
        assert len(sol.t) == len(t_eval)
        assert np.allclose(sol.t, t_eval)
    
    def test_multiple_components(self):
        """Test solving with multiple dependent variables."""
        def f(t, y):
            return [-y[0], 2*y[1]]  # y1' = -y1, y2' = 2*y2
        
        t_span = (0, 1)
        y0 = [1.0, 1.0]
        
        sol = solve(f, t_span, y0, dense_output=True)
        
        assert isinstance(sol, Solution)
        # y1 should decay
        assert sol.y[-1][0] < 1.0
        # y2 should grow
        assert sol.y[-1][1] > 1.0
    
    def test_system_with_second_order_conversion(self):
        """Test solve with a second-order system converted to first order."""
        # Solve y'' = -y (harmonic oscillator)
        def f(t, y, v):
            return -y
        
        system = second_order(f)
        
        t_span = (0, 2*np.pi)
        y0 = [1.0, 0.0]  # Initial position 1, velocity 0
        
        sol = solve(system, t_span, y0, dense_output=True, max_step=0.1)
        
        assert isinstance(sol, Solution)
        # Solution should oscillate back to approximately initial position
        assert np.isclose(sol.y[-1][0], 1.0, atol=0.1)
    
    def test_linear_system_solve(self):
        """Test solve with a linear system."""
        A = [[-1, 0], [0, -2]]
        system = linear(A)
        
        t_span = (0, 1)
        y0 = [1.0, 1.0]
        
        sol = solve(system, t_span, y0, dense_output=True)
        
        assert isinstance(sol, Solution)
        # y1 should decay as e^(-t)
        assert np.isclose(sol.y[-1][0], np.exp(-1), atol=0.01)
        # y2 should decay as e^(-2t)
        assert np.isclose(sol.y[-1][1], np.exp(-2), atol=0.01)
    
    def test_different_solver_methods(self):
        """Test solve with different integration methods."""
        def f(t, y):
            return -y
        
        t_span = (0, 1)
        y0 = [1.0]
        
        # Test RK45 (default)
        sol_rk45 = solve(f, t_span, y0, method="RK45")
        
        # Test RK23
        sol_rk23 = solve(f, t_span, y0, method="RK23")
        
        # Both should give similar results
        assert isinstance(sol_rk45, Solution)
        assert isinstance(sol_rk23, Solution)
        assert np.isclose(sol_rk45.y[-1][0], sol_rk23.y[-1][0], atol=0.001)
    
    def test_negative_time_step(self):
        """Test solving ODEs backwards in time."""
        def f(t, y):
            return -y
        
        # Forward: y(t) = e^(-t)
        # Backward from t=1 to t=0 should give us back y(0) = 1
        t_span = (1, 0)
        y0 = [np.exp(-1)]
        
        sol = solve(f, t_span, y0, dense_output=True)
        
        assert isinstance(sol, Solution)
        # Should return close to y(0) = 1
        assert np.isclose(sol.y[-1][0], 1.0, atol=0.01)
    
    def test_solution_final_method(self):
        """Test Solution.final() method returns last state."""
        def f(t, y):
            return y
        
        t_span = (0, 1)
        y0 = [1.0, 2.0]
        
        sol = solve(f, t_span, y0, dense_output=True)
        
        final = sol.final()
        assert np.allclose(final, sol.y[-1])
    
    def test_stiff_ode(self):
        """Test solving a stiff ODE (needs appropriate method)."""
        # Van der Pol oscillator is moderately stiff
        def f(t, y):
            return [y[1], 100*(1 - y[0]**2)*y[1] - y[0]]
        
        t_span = (0, 10)
        y0 = [2.0, 0.0]
        
        sol = solve(f, t_span, y0, method="RK45", max_step=0.1)
        
        assert isinstance(sol, Solution)
        assert len(sol.y) > 1
    
    def test_with_vector_initial_condition(self):
        """Test solve with Vector as initial condition."""
        def f(t, y):
            return -y
        
        t_span = (0, 1)
        y0 = Vector([1.0, 2.0, 3.0])
        
        sol = solve(f, t_span, y0, dense_output=True)
        
        assert isinstance(sol, Solution)
        # Check that all 3 components are solved
        assert sol.y.shape[1] >= 3 or len(sol.y[0]) == 3


class TestODEIntegration:
    """Integration tests combining multiple ODE components."""
    
    def test_simple_system(self):
        """Test solving a simple coupled system of ODEs."""
        # Simple coupled system: both decay
        def f(t, y):
            x, y = y
            return [-x, -2*y]
        
        t_span = (0, 1)
        y0 = [10.0, 5.0]
        
        sol = solve(f, t_span, y0, t_eval=np.linspace(0, 1, 10))
        
        assert isinstance(sol, Solution)
        assert len(sol.t) == 10
        # Initial values should be above final values
        assert sol.y[0, 0] > sol.y[-1, 0]
        assert sol.y[0, 1] > sol.y[-1, 1]
    
    def test_time_dependent_ode(self):
        """Test solving an ODE with explicit time dependence."""
        def f(t, y):
            return t * y  # y' = t*y
        
        t_span = (0, 1)
        y0 = [1.0]
        
        sol = solve(f, t_span, y0, t_eval=np.linspace(0, 1, 10))
        
        assert isinstance(sol, Solution)
        # y(t) = exp(t^2/2)
        # At t=1: y = exp(0.5)
        assert np.isclose(sol.y[-1][0], np.exp(0.5), atol=0.01)
    
    def test_solve_with_matrix_linear_system(self):
        """Test solving a linear system created with Matrix."""
        A_data = [[0, 1], [-1, 0]]  # Rotation
        A = Matrix(A_data)
        
        # Manually create system function from matrix
        def system(t, x):
            result = A @ Vector(x)
            return result.data
        
        t_span = (0, 2*np.pi)
        y0 = [1.0, 0.0]
        
        sol = solve(system, t_span, y0, max_step=0.1)
        
        assert isinstance(sol, Solution)
        # After full rotation, should return to start
        assert np.isclose(sol.y[-1][0], 1.0, atol=0.1)
        assert np.isclose(sol.y[-1][1], 0.0, atol=0.1)


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_single_point_evaluation(self):
        """Test with minimal time span."""
        def f(t, y):
            return -y
        
        t_span = (0, 0.001)
        y0 = [1.0]
        
        sol = solve(f, t_span, y0)
        
        assert isinstance(sol, Solution)
        assert len(sol.t) >= 1
    
    def test_zero_initial_condition(self):
        """Test with zero initial condition."""
        def f(t, y):
            return y
        
        t_span = (0, 1)
        y0 = [0.0]
        
        sol = solve(f, t_span, y0)
        
        assert isinstance(sol, Solution)
        assert np.allclose(sol.y, 0.0)
    
    def test_large_initial_condition(self):
        """Test with large initial values."""
        def f(t, y):
            return -y
        
        t_span = (0, 1)
        y0 = [1e6]
        
        sol = solve(f, t_span, y0)
        
        assert isinstance(sol, Solution)
        assert sol.y[-1][0] < y0[0]
    
    def test_negative_initial_values(self):
        """Test with negative initial conditions."""
        def f(t, y):
            return -y
        
        t_span = (0, 1)
        y0 = [-5.0]
        
        sol = solve(f, t_span, y0)
        
        assert isinstance(sol, Solution)
        assert sol.y[-1][0] > y0[0]  # Less negative after decay
        assert sol.y[-1][0] < 0  # Still negative
