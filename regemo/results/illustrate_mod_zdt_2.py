import matplotlib.pyplot as plt
import numpy as np


def draw_circle():
    np.random.seed(0)
    x1 = np.random.rand(10000)
    x1 = np.sort(x1)
    x2 = np.sqrt(1 - x1 ** 2)
    fig, ax = plt.subplots()
    ax.plot(x1, x2, color="blue", label="original rule")
    ax.annotate("$x_1^2 + x_2^2 = 1$", xy=(0.42, 0.82), fontsize=15, rotation=-23, color="blue")
    ax.set_xlabel("$x_1$", fontsize=15)
    ax.set_ylabel("$x_2$", fontsize=15)
    # ax.grid(alpha=0.3)
    return fig, ax


def draw_rule():
    fig, ax = draw_circle()
    np.random.seed(0)
    x1 = np.random.rand(10000)
    x1 = np.sort(x1)
    x2 = -0.6 * x1 + 1.07
    ax.plot(x1, x2, color="red", label="regular rule")
    ax.annotate("$x_2 + 0.6 x_1 = 1.07$", xy=(0.3, 0.6), fontsize=15, rotation=-23, color="red")
    ax.legend(loc="lower left", fontsize=15)
    fig.savefig("mod_zdt_2d_illustration.eps", dpi=400)
    fig.show()


def main():
    draw_rule()


if __name__ == "__main__":
    main()