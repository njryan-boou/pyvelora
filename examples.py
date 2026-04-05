"""Examples demonstrating the current pyvelora API."""

from __future__ import annotations

import math

import pyvelora as pv


def section(title: str) -> None:
	print("\n" + "=" * 70)
	print(title)
	print("=" * 70)


def main() -> None:
	section("VECTOR OPERATIONS")
	v1 = pv.Vector([1, 2, 3])
	v2 = pv.Vector([4, 5, 6])
	print("v1 =", v1)
	print("v2 =", v2)
	print("v1 + v2 =", v1 + v2)
	print("v2 - v1 =", v2 - v1)
	print("dot(v1, v2) =", pv.linalg.dot(v1, v2))
	print("||v1||_2 =", pv.linalg.vector_norm(v1))
	print("v1 x v2 =", pv.linalg.cross(v1, v2))

	section("MATRIX OPERATIONS")
	A = pv.Matrix([[1, 2], [3, 4]])
	B = pv.Matrix([[2, 0], [1, 2]])
	print("A =\n", A)
	print("B =\n", B)
	print("A + B =\n", A + B)
	print("A @ B =\n", A @ B)
	print("trace(A) =", pv.linalg.trace(A))
	print("det(A) =", pv.linalg.determinant(A))
	print("inverse(A) =\n", pv.linalg.inverse(A))

	section("DECOMPOSITIONS")
	M = pv.Matrix([[4, 3], [6, 3]])
	L, U = pv.linalg.lu_decomposition(M)
	Q, R = pv.linalg.qr_decomposition(M)
	print("M =\n", M)
	print("L =\n", L)
	print("U =\n", U)
	print("L @ U =\n", L @ U)
	print("Q =\n", Q)
	print("R =\n", R)
	print("Q @ R =\n", Q @ R)

	section("LINEAR SYSTEMS")
	A_sys = pv.Matrix([[3, 1], [1, 2]])
	b_sys = pv.Vector([9, 8])
	x = pv.linalg.solve_linear_system(A_sys, b_sys)
	print("Solve A x = b")
	print("A =\n", A_sys)
	print("b =", b_sys)
	print("x =", x)
	print("A @ x =", A_sys @ x)

	section("TENSOR OPERATIONS")
	t1 = pv.Tensor([
		[[1, 2], [3, 4]],
		[[5, 6], [7, 8]],
	])
	t2 = pv.Tensor([
		[[2, 1], [0, 1]],
		[[1, 0], [2, 1]],
	])
	print("t1.shape =", t1.shape)
	print("t2.shape =", t2.shape)
	print("(t1 + t2).shape =", (t1 + t2).shape)
	print("t1.transpose((1, 0, 2)).shape =", t1.transpose((1, 0, 2)).shape)
	print("t1.kron(t2).shape =", t1.kron(t2).shape)

	section("UTILITIES")
	print("clean([1e-15, 1.0, -1e-14]) =", pv.utils.clean([1e-15, 1.0, -1e-14]))
	print("allclose([1, 2], [1, 2.0000001]) =", pv.utils.allclose([1.0, 2.0], [1.0, 2.0000001]))
	print("linspace(0, 1, 5) =", pv.utils.linspace(0.0, 1.0, 5))
	print("logspace(0, 2, 3) =", pv.utils.logspace(0.0, 2.0, 3))

	section("DIFFYQ")
	sol = pv.diffyq.solve_linear(
		A=[[0.0, 1.0], [-1.0, 0.0]],
		x0=[1.0, 0.0],
		t0=0.0,
		tf=2.0 * math.pi,
		t_eval=[0.0, math.pi / 2.0, math.pi, 3.0 * math.pi / 2.0, 2.0 * math.pi],
	)
	print("solve_linear success =", sol.success)
	print("final state =", sol.final)

	section("END OF EXAMPLES")


if __name__ == "__main__":
	main()
