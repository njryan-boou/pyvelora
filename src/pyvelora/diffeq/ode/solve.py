from __future__ import annotations
import numpy as np
from scipy.integrate import solve_ivp
from .solution import Solution


def solve(f, t_span, y0, *, t_eval=None, method="RK45", **kwargs) -> Solution:
    
    
    y0 = np.asarray(y0, dtype=float)

    sol = solve_ivp(
        f,
        t_span,
        y0,
        t_eval=t_eval,
        method=method,
        **kwargs
    )

    return Solution(sol.t, sol.y.T)