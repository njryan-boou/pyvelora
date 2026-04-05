from scipy.integrate import solve_ivp
from pyvelora.diffyq.utils import _to_list, _from_list


class ODESolution:
    def __init__(self, sol, y0):
        self._sol = sol
        self.t = sol.t
        self.y = sol.y
        self.success = sol.success
        self.message = sol.message
        self._y0 = y0

    @property
    def final(self):
        return _from_list(self._y0, self.y[:, -1])

    def at(self, i):
        return _from_list(self._y0, self.y[:, i])

    def pairs(self):
        return [(self.t[i], self.at(i)) for i in range(len(self.t))]


def solve_ivp_wrapper(
    f,
    t_span,
    y0,
    t_eval=None,
    method="RK45",
    rtol=1e-6,
    atol=1e-9,
    **kwargs
):
    y0_list = _to_list(y0)

    sol = solve_ivp(
        fun=f,
        t_span=t_span,
        y0=y0_list,
        t_eval=t_eval,
        method=method,
        rtol=rtol,
        atol=atol,
        **kwargs
    )

    return ODESolution(sol, y0)