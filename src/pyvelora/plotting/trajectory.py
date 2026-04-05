import matplotlib.pyplot as plt
from pyvelora.plotting.utils import setup_axes


def trajectory(sol):
    """
    Plot trajectory in phase space (2D only)
    """
    y = sol.y

    if y.shape[0] != 2:
        raise ValueError("Trajectory plot only supports 2D systems")

    x, v = y

    setup_axes()
    plt.plot(x, v)
    plt.xlabel("x")
    plt.ylabel("v")
    plt.title("Trajectory")
    plt.show()