from __future__ import annotations

from typing import Any


# Default absolute tolerance used for zero checks and cleanup.
zero_threshold: float = 1e-10


def set_precision(*, zero_tol: float | None = None) -> None:
	"""Update global precision settings for utility helpers."""
	global zero_threshold

	if zero_tol is not None:
		if zero_tol < 0:
			raise ValueError("zero_tol must be non-negative")
		zero_threshold = float(zero_tol)


def get_precision() -> dict[str, float]:
	"""Return current precision settings."""
	return {"zero_threshold": zero_threshold}


def isclose(a: Any, b: Any, *, rtol: float = 1e-5, atol: float | None = None) -> Any:
	"""Elementwise closeness check. Handles scalars and arbitrarily nested lists."""
	if atol is None:
		atol = zero_threshold

	# Delegate to the object's own implementation when available (e.g. numpy arrays).
	if hasattr(a, "__abs__") and not isinstance(a, (int, float, complex, list)):
		threshold = atol + rtol * abs(b)
		return abs(a - b) <= threshold

	if isinstance(a, (int, float, complex)):
		threshold = atol + rtol * abs(b)
		return abs(a - b) <= threshold

	# Nested list / list-like: recurse element-wise.
	a_iter = list(a)
	b_iter = b if isinstance(b, (list, tuple)) else [b] * len(a_iter)
	return [isclose(ai, bi, rtol=rtol, atol=atol) for ai, bi in zip(a_iter, b_iter)]


def allclose(a: Any, b: Any, *, rtol: float = 1e-5, atol: float | None = None) -> bool:
	"""Aggregate closeness check that returns a single bool."""
	if atol is None:
		atol = zero_threshold
	result = isclose(a, b, rtol=rtol, atol=atol)
	if isinstance(result, bool):
		return result
	if hasattr(result, "all"):
		return bool(result.all())

	def _all_flat(v: Any) -> bool:
		if isinstance(v, list):
			return all(_all_flat(item) for item in v)
		return bool(v)

	return _all_flat(result)


def is_zero(x: Any, *, atol: float | None = None) -> Any:
	"""Elementwise zero check. Works on scalars and nested lists."""
	if atol is None:
		atol = zero_threshold
	return isclose(x, 0.0, atol=atol, rtol=0.0)


def is_close(a: Any, b: Any, *, rtol: float = 1e-5, atol: float | None = None) -> Any:
	"""Alias for isclose."""
	return isclose(a, b, rtol=rtol, atol=atol)


def round_small(x: Any, *, atol: float | None = None) -> Any:
	"""Set numerically small values to exact zero, preserving nested structure."""
	if atol is None:
		atol = zero_threshold

	if isinstance(x, (int, float, complex)):
		return 0.0 if abs(x) <= atol else x

	if isinstance(x, list):
		return [round_small(item, atol=atol) for item in x]

	# Non-list iterables (e.g. custom data wrappers): iterate and return a list.
	return [round_small(item, atol=atol) for item in x]


def clean(x: Any, *, atol: float | None = None) -> Any:
	"""Normalize tiny floating artifacts to zero."""
	return round_small(x, atol=atol)


def round_to(x: Any, digits: int) -> Any:
	"""Round x to the given number of decimal places, preserving nested structure."""
	if not isinstance(digits, int) or digits < 0:
		raise ValueError("digits must be a non-negative integer")

	if isinstance(x, (int, float)):
		return round(x, digits)

	if isinstance(x, complex):
		return complex(round(x.real, digits), round(x.imag, digits))

	if isinstance(x, list):
		return [round_to(item, digits) for item in x]

	return [round_to(item, digits) for item in x]


def is_integer(x: Any, *, atol: float | None = None) -> Any:
	"""Check whether x is numerically equal to an integer, elementwise."""
	if atol is None:
		atol = zero_threshold

	if isinstance(x, (int, float, complex)):
		return abs(x - round(x.real if isinstance(x, complex) else x)) <= atol

	if isinstance(x, list):
		return [is_integer(item, atol=atol) for item in x]

	return [is_integer(item, atol=atol) for item in x]


__all__ = [
	"zero_threshold",
	"set_precision",
	"get_precision",
	"clean",
	"isclose",
	"allclose",
	"is_zero",
	"is_close",
	"is_integer",
	"round_small",
	"round_to",
]





