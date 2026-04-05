import matplotlib.pyplot as plt


def solution(sol, labels=None):
    """
    Plot solution vs time
    """
    t = sol.t
    y = sol.y

    if y.ndim == 1 or y.shape[0] == 1:
        plt.plot(t, y[0] if y.ndim > 1 else y)
    else:
        for i in range(y.shape[0]):
            label = labels[i] if labels else f"x{i}"
            plt.plot(t, y[i], label=label)
        plt.legend()

    plt.xlabel("t")
    plt.ylabel("y")
    plt.title("Solution")
    plt.grid()
    plt.show()