from pyvelora.diffyq.system import solve_system


def solve_second_order(accel, t0, y0, v0, tf, t_eval=None):
    def system(t, state):
        y, v = state
        return [v, accel(t, y, v)]

    return solve_system(
        system,
        t0,
        [y0, v0],
        tf,
        t_eval=t_eval
    )