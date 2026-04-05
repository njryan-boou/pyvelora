"""Tests that all public symbols can be imported from pyvelora."""

# ---------------------------------------------------------------------------
# Top-level package
# ---------------------------------------------------------------------------

def test_top_level_core_types():
    from pyvelora import Vector, Matrix, Tensor
    assert callable(Vector)
    assert callable(Matrix)
    assert callable(Tensor)


def test_top_level_linalg_module():
    import pyvelora
    assert hasattr(pyvelora, "linalg")


# ---------------------------------------------------------------------------
# pyvelora.core
# ---------------------------------------------------------------------------

def test_core_submodules():
    from pyvelora.core import vector, matrix, tensor, array_base


def test_core_symbols():
    from pyvelora.core import Vector, Matrix, Tensor, Base, format_array


def test_core_vector_module():
    from pyvelora.core.vector import Vector
    assert callable(Vector)


def test_core_matrix_module():
    from pyvelora.core.matrix import Matrix
    assert callable(Matrix)


def test_core_tensor_module():
    from pyvelora.core.tensor import Tensor
    assert callable(Tensor)


def test_core_array_base_module():
    from pyvelora.core.array_base import Base, format_array
    assert callable(Base)
    assert callable(format_array)


# ---------------------------------------------------------------------------
# pyvelora.linalg
# ---------------------------------------------------------------------------

def test_linalg_decomposition_functions():
    from pyvelora.linalg import lu_decomposition, qr_decomposition, svd_decomposition
    assert callable(lu_decomposition)
    assert callable(qr_decomposition)
    assert callable(svd_decomposition)


def test_linalg_constructor_functions():
    from pyvelora.linalg import zeros, ones, full, identity, diagonal, from_rows, from_cols
    assert callable(zeros)
    assert callable(ones)
    assert callable(full)
    assert callable(identity)
    assert callable(diagonal)
    assert callable(from_rows)
    assert callable(from_cols)


def test_linalg_check_functions():
    from pyvelora.linalg import (
        is_square, is_symmetric, is_orthogonal, is_singular, is_invertible,
        is_diagonal, is_identity, is_upper_triangular, is_lower_triangular, is_rref,
        is_skew_symmetric, is_positive_definite,
    )
    assert callable(is_square)


def test_linalg_norm_functions():
    from pyvelora.linalg import vector_norm, frobenius_norm, one_norm, inf_norm, normalize
    assert callable(vector_norm)


def test_linalg_product_functions():
    from pyvelora.linalg import dot, outer, matmul, matvec, cross, prod
    assert callable(dot)


def test_linalg_property_functions():
    from pyvelora.linalg import trace, rank, determinant, inverse, adjugate
    assert callable(trace)


def test_linalg_solve_functions():
    from pyvelora.linalg import forward_substitution, backward_substitution, solve_linear_system
    assert callable(solve_linear_system)


def test_linalg_eigen_functions():
    from pyvelora.linalg import eigenvalues, eigenvectors
    assert callable(eigenvalues)


def test_linalg_rref():
    from pyvelora.linalg import rref
    assert callable(rref)


def test_linalg_matrix_functions():
    from pyvelora.linalg import matrix_exponential, matrix_power
    assert callable(matrix_exponential)


def test_linalg_basic_functions():
    from pyvelora.linalg import get_row, get_col, swap_rows, swap_cols, transpose
    assert callable(get_row)


# ---------------------------------------------------------------------------
# pyvelora.linalg — subspaces and additional decompositions
# ---------------------------------------------------------------------------

def test_linalg_subspace_functions():
    from pyvelora.linalg import column_space, row_space, null_space, left_null_space
    assert callable(column_space)
    assert callable(row_space)
    assert callable(null_space)
    assert callable(left_null_space)


def test_linalg_extra_decomposition_functions():
    from pyvelora.linalg import eigen_decomposition, cholesky_decomposition, schur_decomposition
    assert callable(eigen_decomposition)
    assert callable(cholesky_decomposition)
    assert callable(schur_decomposition)


# ---------------------------------------------------------------------------
# pyvelora.constants
# ---------------------------------------------------------------------------

def test_constants_math():
    from pyvelora.constants import pi, tau, e, golden_ratio, euler_mascheroni, inf, nan
    assert isinstance(pi, float)


def test_constants_physical():
    from pyvelora.constants import c, h, hbar, G, q_e, k_B, N_A, R
    assert c == 299_792_458.0


# ---------------------------------------------------------------------------
# Top-level submodule exposure
# ---------------------------------------------------------------------------

def test_top_level_exposes_constants():
    import pyvelora
    assert hasattr(pyvelora, "constants")


def test_top_level_exposes_diffyq():
    import pyvelora
    assert hasattr(pyvelora, "diffyq")


def test_top_level_exposes_utils():
    import pyvelora
    assert hasattr(pyvelora, "utils")


def test_top_level_exposes_plotting():
    import pyvelora
    assert hasattr(pyvelora, "plotting")


def test_linalg_submodules_direct():
    from pyvelora.linalg import basic, checks, constructors, decompositions
    from pyvelora.linalg import matrix_functions, norms, products, properties, rref, solve, eigen


# ---------------------------------------------------------------------------
# pyvelora.utils
# ---------------------------------------------------------------------------

def test_utils_submodules():
    from pyvelora.utils import errors, numpy_utils, precision, validation


def test_utils_error_classes():
    from pyvelora.utils import PyveloraError, ShapeError, DimensionError
    assert issubclass(ShapeError, PyveloraError)
    assert issubclass(DimensionError, PyveloraError)


def test_utils_numpy_utils():
    from pyvelora.utils import meshgrid, linspace_nd, logspace_nd, arange_nd
    assert callable(meshgrid)
    assert callable(linspace_nd)
    assert callable(logspace_nd)
    assert callable(arange_nd)


def test_utils_precision():
    from pyvelora.utils import (
        clean, isclose, allclose, set_precision, get_precision,
        is_zero, is_close, round_small,
    )
    assert callable(clean)
    assert callable(isclose)
    assert callable(allclose)
    assert callable(set_precision)
    assert callable(get_precision)


def test_utils_validation():
    from pyvelora.utils import (
        require_vector, require_matrix, require_tensor,
        require_same_shape, require_square, require_dimension,
    )
    assert callable(require_vector)
    assert callable(require_matrix)
    assert callable(require_tensor)


def test_utils_errors_module_direct():
    from pyvelora.utils.errors import PyveloraError, ShapeError, DimensionError


def test_utils_precision_module_direct():
    from pyvelora.utils.precision import clean, isclose, allclose, set_precision, get_precision


def test_utils_validation_module_direct():
    from pyvelora.utils.validation import require_vector, require_matrix, require_tensor


def test_utils_numpy_utils_module_direct():
    from pyvelora.utils.numpy_utils import meshgrid, linspace_nd, logspace_nd, arange_nd


# ---------------------------------------------------------------------------
# pyvelora.constants
# ---------------------------------------------------------------------------

def test_constants_submodule():
    from pyvelora.constants import constants


def test_constants_symbols():
    from pyvelora.constants import pi, e, golden_ratio


def test_constants_values():
    import math
    from pyvelora.constants import pi, e, golden_ratio
    assert abs(pi - math.pi) < 1e-10
    assert abs(e - math.e) < 1e-10
    assert abs(golden_ratio - (1 + math.sqrt(5)) / 2) < 1e-10


def test_constants_module_direct():
    from pyvelora.constants.constants import pi, e, golden_ratio


# ---------------------------------------------------------------------------
# pyvelora.diffyq
# ---------------------------------------------------------------------------

def test_diffyq_symbols():
    from pyvelora.diffyq import ODESolution, solve_ivp_wrapper, solve_system
    from pyvelora.diffyq import solve_linear, solve_second_order
    assert callable(solve_ivp_wrapper)
    assert callable(solve_system)
    assert callable(solve_linear)
    assert callable(solve_second_order)


def test_diffyq_submodules_direct():
    from pyvelora.diffyq import ivp, system, linear_system, second_order, utils


def test_diffyq_ivp_module():
    from pyvelora.diffyq.ivp import ODESolution, solve_ivp_wrapper
    assert callable(solve_ivp_wrapper)


def test_diffyq_system_module():
    from pyvelora.diffyq.system import solve_system
    assert callable(solve_system)


def test_diffyq_linear_system_module():
    from pyvelora.diffyq.linear_system import solve_linear
    assert callable(solve_linear)


def test_diffyq_second_order_module():
    from pyvelora.diffyq.second_order import solve_second_order
    assert callable(solve_second_order)


# ---------------------------------------------------------------------------
# pyvelora.plotting
# ---------------------------------------------------------------------------

def test_plotting_symbols():
    from pyvelora.plotting import solution, trajectory, vector_field, phase_portrait
    assert callable(solution)
    assert callable(trajectory)
    assert callable(vector_field)
    assert callable(phase_portrait)


def test_plotting_submodules_direct():
    from pyvelora.plotting import phase, utils
    from pyvelora.plotting.solution import solution
    from pyvelora.plotting.trajectory import trajectory
    from pyvelora.plotting.vector_field import vector_field
    from pyvelora.plotting.phase import phase_portrait


