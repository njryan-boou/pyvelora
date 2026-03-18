import numpy as np
import matplotlib.pyplot as plt


def phase_portrait(f, x_range, y_range, density=20):

    x = np.linspace(*x_range, density)
    y = np.linspace(*y_range, density)

    X, Y = np.meshgrid(x, y)

    U = np.zeros_like(X)
    V = np.zeros_like(Y)

    for i in range(density):
        for j in range(density):

            dx, dy = f(0, [X[i,j], Y[i,j]])

            U[i,j] = dx
            V[i,j] = dy

    plt.streamplot(X, Y, U, V)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Phase Portrait")

    plt.show()