import matplotlib.pyplot as plt


def solution_plot(sol):

    plt.plot(sol.t, sol.y)

    plt.xlabel("t")
    plt.ylabel("state")
    plt.title("Solution")

    plt.show()