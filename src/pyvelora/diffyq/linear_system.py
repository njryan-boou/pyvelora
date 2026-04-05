def solve_linear(A, x0, t0, tf, t_eval=None, method="RK45"):
    coeffs = getattr(A, "data", A)
    n = len(coeffs)

    if any(len(row) != n for row in coeffs):
        raise ValueError("A must be a square matrix-like object")

    def system(t, x):
        return [
            sum(coeffs[i][j] * x[j] for j in range(n))
            for i in range(n)
        ]

    from pyvelora.diffyq.system import solve_system

    return solve_system(
        system,
        t0,
        x0,
        tf,
        t_eval=t_eval,
        method=method
    )