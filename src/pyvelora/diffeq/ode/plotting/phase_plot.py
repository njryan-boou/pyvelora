import matplotlib.pyplot as plt


def phase_plot(sol):

    y = sol.y

    if y.shape[1] < 2:
        raise ValueError("Phase plot requires at least 2 variables.")

    plt.plot(y[:,0], y[:,1])

    plt.xlabel("x")
    plt.ylabel("v")
    plt.title("Phase Plot")

    plt.show()