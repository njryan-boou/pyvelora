
import matplotlib.pyplot as plt
from pyvelora.plotting.utils import setup_axes
from pyvelora.utils.numpy_utils import linspace, meshgrid

def vector_field(F, x_range=(-5,5), y_range=(-5,5), density=20):
    """
    Plot vector field for system dx/dt = F(x)
    """
    x = linspace(*x_range, density)
    y = linspace(*y_range, density)

    X, Y = meshgrid(x, y)

    U = [[0.0 for _ in x] for _ in y]
    V = [[0.0 for _ in x] for _ in y]

    for i in range(density):
        for j in range(density):
            dx, dy = F([X[i][j], Y[i][j]])
            U[i][j] = dx
            V[i][j] = dy

    setup_axes()
    plt.quiver(X, Y, U, V)
    plt.title("Vector Field")
    plt.show()