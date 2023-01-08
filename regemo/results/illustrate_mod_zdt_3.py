import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl


def draw_sphere():
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    x2 = np.outer(np.linspace(0, 1, 10000), np.ones(10000))
    x3 = x2.copy().T
    x1 = np.sqrt(1 - x2 ** 2 - x3 ** 2)
    ax.view_init(20, -90)
    ax.plot_surface(x1, x2, x3, color="blue", alpha=0.3, label=f"original rule: $x_1^2 + x_2^2 + x_3^2 = 1$")
    ax.set_xlabel("$x_1$", fontsize=15)
    ax.set_ylabel("$x_2$", fontsize=15)
    ax.set_zlabel("$x_3$", fontsize=15)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    return fig, ax


def draw_rule():
    fig, ax = draw_sphere()
    np.random.seed(0)
    x2 = np.outer(np.linspace(0, 1, 100), np.ones(100))
    x3 = x2.copy().T
    x1 = -x3 - (0.5 * x2) + 1.23
    ax.plot_surface(x1, x2, x3, color="red", alpha=0.3, label="original rule: $x_1 + 0.5x_2 + x_3 = 1.23$")
    fake2Dline_1 = mpl.lines.Line2D([0], [0], linestyle="none", c='b', marker='o')
    fake2Dline_2 = mpl.lines.Line2D([0], [0], linestyle="none", c='r', marker='o')
    ax.legend([fake2Dline_1, fake2Dline_2], ['original rule: $x_1^2 + x_2^2 + x_3^2 = 1$', "regular rule: "
                                                                                           "$x_1 + 0.5x_2 + x_3 = 1.23$"], numpoints=1)
    plt.show()
    plt.show()



def main():
    draw_rule()


if __name__ == "__main__":
    main()