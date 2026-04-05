from __future__ import annotations

from importlib import import_module


_MODULES = {
	"basic": "pyvelora.linalg.basic",
	"checks": "pyvelora.linalg.checks",
	"constructors": "pyvelora.linalg.constructors",
	"decompositions": "pyvelora.linalg.decompositions",
	"subspaces": "pyvelora.linalg.subspaces",
	"matrix_functions": "pyvelora.linalg.matrix_functions",
	"norms": "pyvelora.linalg.norms",
	"products": "pyvelora.linalg.products",
	"properties": "pyvelora.linalg.properties",
	"rref": "pyvelora.linalg.rref",
	"solve": "pyvelora.linalg.solve",
	"eigen": "pyvelora.linalg.eigen",
}

_SYMBOLS = {
	"get_row": "pyvelora.linalg.basic",
	"get_col": "pyvelora.linalg.basic",
	"swap_rows": "pyvelora.linalg.basic",
	"swap_cols": "pyvelora.linalg.basic",
	"transpose": "pyvelora.linalg.basic",
	"add": "pyvelora.linalg.basic",
	"subtract": "pyvelora.linalg.basic",
	"scalar_multiply": "pyvelora.linalg.basic",
	"hadamard_product": "pyvelora.linalg.basic",
	"hamard_product": "pyvelora.linalg.basic",
	"is_square": "pyvelora.linalg.checks",
	"is_symmetric": "pyvelora.linalg.checks",
	"is_orthogonal": "pyvelora.linalg.checks",
	"is_singular": "pyvelora.linalg.checks",
	"is_invertible": "pyvelora.linalg.checks",
	"is_diagonal": "pyvelora.linalg.checks",
	"is_identity": "pyvelora.linalg.checks",
	"is_skew_symmetric": "pyvelora.linalg.checks",
	"is_positive_definite": "pyvelora.linalg.checks",
	"is_upper_triangular": "pyvelora.linalg.checks",
	"is_lower_triangular": "pyvelora.linalg.checks",
	"is_rref": "pyvelora.linalg.checks",
	"zeros": "pyvelora.linalg.constructors",
	"ones": "pyvelora.linalg.constructors",
	"full": "pyvelora.linalg.constructors",
	"identity": "pyvelora.linalg.constructors",
	"diagonal": "pyvelora.linalg.constructors",
	"from_rows": "pyvelora.linalg.constructors",
	"from_cols": "pyvelora.linalg.constructors",
	"lu_decomposition": "pyvelora.linalg.decompositions",
	"qr_decomposition": "pyvelora.linalg.decompositions",
	"svd_decomposition": "pyvelora.linalg.decompositions",
	"polar_decomposition": "pyvelora.linalg.decompositions",
	"eigen_decomposition": "pyvelora.linalg.decompositions",
	"cholesky_decomposition": "pyvelora.linalg.decompositions",
	"schur_decomposition": "pyvelora.linalg.decompositions",
	"column_space": "pyvelora.linalg.subspaces",
	"row_space": "pyvelora.linalg.subspaces",
	"null_space": "pyvelora.linalg.subspaces",
	"left_null_space": "pyvelora.linalg.subspaces",
	"matrix_exponential": "pyvelora.linalg.matrix_functions",
	"matrix_power": "pyvelora.linalg.matrix_functions",
	"vector_norm": "pyvelora.linalg.norms",
	"frobenius_norm": "pyvelora.linalg.norms",
	"one_norm": "pyvelora.linalg.norms",
	"inf_norm": "pyvelora.linalg.norms",
	"normalize": "pyvelora.linalg.norms",
	"dot": "pyvelora.linalg.products",
	"outer": "pyvelora.linalg.products",
	"matmul": "pyvelora.linalg.products",
	"matvec": "pyvelora.linalg.products",
	"cross": "pyvelora.linalg.products",
	"prod": "pyvelora.linalg.products",
	"trace": "pyvelora.linalg.properties",
	"rank": "pyvelora.linalg.properties",
	"minor": "pyvelora.linalg.properties",
	"cofactor": "pyvelora.linalg.properties",
	"cofactor_matrix": "pyvelora.linalg.properties",
	"adjugate": "pyvelora.linalg.properties",
	"determinant": "pyvelora.linalg.properties",
	"inverse": "pyvelora.linalg.properties",
	"rref": "pyvelora.linalg.rref",
	"forward_substitution": "pyvelora.linalg.solve",
	"backward_substitution": "pyvelora.linalg.solve",
	"solve_lu": "pyvelora.linalg.solve",
	"solve_linear_system": "pyvelora.linalg.solve",
	"eigenvalues": "pyvelora.linalg.eigen",
	"eigenvectors": "pyvelora.linalg.eigen",
}

__all__ = list(_MODULES) + list(_SYMBOLS)


_COLLIDING_EXPORTS = set(_MODULES) & set(_SYMBOLS)


def _restore_colliding_symbols() -> None:
	for name in _COLLIDING_EXPORTS:
		module = import_module(_SYMBOLS[name])
		globals()[name] = getattr(module, name)


def __getattr__(name: str) -> object:
	if name in _SYMBOLS:
		module = import_module(_SYMBOLS[name])
		value = getattr(module, name)
		globals()[name] = value
		_restore_colliding_symbols()
		return value
	if name in _MODULES:
		module = import_module(_MODULES[name])
		globals()[name] = module
		_restore_colliding_symbols()
		return module
	raise AttributeError(f"module 'pyvelora.linalg' has no attribute {name!r}")