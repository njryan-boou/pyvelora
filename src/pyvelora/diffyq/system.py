from pyvelora.diffyq.ivp import solve_ivp_wrapper


def solve_system(
    F,
    t0,
    x0,
    tf,
    t_eval=None,
    method="RK45",
    **kwargs
):
    return solve_ivp_wrapper(
        f=F,
        t_span=(t0, tf),
        y0=x0,
        t_eval=t_eval,
        method=method,
        **kwargs
    )