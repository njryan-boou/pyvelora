import matplotlib.pyplot as plt


def setup_axes(equal=True, grid=True):
    ax = plt.gca()
    if equal:
        ax.set_aspect('equal')
    if grid:
        plt.grid(True)
    plt.axhline(0)
    plt.axvline(0)
    return ax