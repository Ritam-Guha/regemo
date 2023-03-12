import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

from regemo.problems.get_problem import zdt_mod_2d


def fit_linreg(points,
               deg=2):
    x = points[:, 0]
    y = points[:, 1]
    X = np.column_stack((x, x**2))
    fit = LinearRegression().fit(X, y)
    return fit


def draw_circle():
    np.random.seed(0)
    x1 = np.random.rand(10000)
    x1 = np.sort(x1)
    x2 = np.sqrt(1 - x1 ** 2)
    fig, ax = plt.subplots()
    ax.plot(x1, x2, color="blue", label="original rule: $x_1^2 + x_2^2 = 1$")
    # ax.annotate("$x_1^2 + x_2^2 = 1$", xy=(0.42, 0.82), fontsize=15, rotation=-23, color="blue")
    ax.set_xlabel("$x_1$", fontsize=15)
    ax.set_ylabel("$x_2$", fontsize=15)
    # ax.grid(alpha=0.3)

    fit = fit_linreg(np.column_stack((x1, x2)))
    x2_reg_non_lin = fit.predict(np.column_stack((x1, x1**2)))
    reg_non_lin_valid = np.where(np.logical_and(0 <= x2_reg_non_lin, x2_reg_non_lin <= 1))
    ax.plot(x1[reg_non_lin_valid], x2_reg_non_lin[reg_non_lin_valid], linestyle="dashed", dashes=(3, 3), linewidth=3, color="black", label=f"regular non-linear rule: $x_2 = {np.round(fit.coef_[0], 2)} x_1  {np.round(fit.coef_[1], 2)} x_1^2 + {np.round(fit.intercept_, 2)}$")
    # ax.annotate("$x_1^2 + x_2^2 = 1$", xy=(0.42, 0.82), fontsize=15, rotation=-23, color="blue")

    return fig, ax


def draw_rule():
    fig, ax = draw_circle()
    np.random.seed(0)
    x1 = np.random.rand(10000)
    x1 = np.sort(x1)
    x2_reg_lin = -0.5 * x1 + 1
    reg_lin_valid = np.where(np.logical_and(0 <= x2_reg_lin, x2_reg_lin <= 1))
    ax.plot(x1[reg_lin_valid], x2_reg_lin[reg_lin_valid], color="red", label="regular linear rule: $x_2 + 0.5 x_1 = 1$")
    # ax.annotate("$x_2 + 0.6 x_1 = 1.07$", xy=(0.3, 0.6), fontsize=15, rotation=-23, color="red")
    ax.legend(loc="lower left", fontsize=10)
    fig.savefig("mod_zdt_2d_illustration_x.pdf", format="pdf")
    fig.show()


def draw_efficient_fronts():
    np.random.seed(0)
    x1 = np.random.rand(10000)
    x2_orig = np.sqrt(1 - x1 ** 2)
    x2_reg_lin = -0.5 * x1 + 1
    fit = fit_linreg(np.column_stack((x1, x2_orig)))
    x2_reg_non_lin = fit.predict(np.column_stack((x1, x1**2)))
    x3 = np.ones(len(x1)) * 0
    x4 = np.ones(len(x1)) * 0.2
    F_orig = zdt_mod_2d.evaluate(np.column_stack((x1, x2_orig, x3, x4)))
    F_reg_lin = zdt_mod_2d.evaluate(np.column_stack((x1, x2_reg_lin, x3, x4)))
    F_reg_non_lin = zdt_mod_2d.evaluate(np.column_stack((x1, x2_reg_non_lin, x3, x4)))

    reg_lin_valid = np.where(np.logical_and(0 <= x2_reg_lin, x2_reg_lin <= 1))
    reg_non_lin_valid = np.where(np.logical_and(0 <= x2_reg_non_lin, x2_reg_non_lin <= 1))

    fig, ax = plt.subplots()
    sort_idx = np.argsort(F_orig[:, 0])
    F_orig[:, 0] = F_orig[sort_idx, 0]
    F_orig[:, 1] = F_orig[sort_idx, 1]
    ax.plot(F_orig[:, 0], F_orig[:, 1], color="blue", label="original efficient front")

    F_reg_non_lin = F_reg_non_lin[reg_non_lin_valid]
    sort_idx = np.argsort(F_reg_non_lin[:, 0])
    F_reg_non_lin[:, 0] = F_reg_non_lin[sort_idx, 0]
    F_reg_non_lin[:, 1] = F_reg_non_lin[sort_idx, 1]
    ax.plot(F_reg_non_lin[:, 0], F_reg_non_lin[:, 1], color="black", label="regular non-linear efficient front", linestyle="dashed", dashes=(3, 3), linewidth=3)

    F_reg_lin = F_reg_lin[reg_lin_valid]
    sort_idx = np.argsort(F_reg_lin[:, 0])
    F_reg_lin[:, 0] = F_reg_lin[sort_idx, 0]
    F_reg_lin[:, 1] = F_reg_lin[sort_idx, 1]
    ax.plot(F_reg_lin[:, 0], F_reg_lin[:, 1], color="red", label="regular linear efficient front")
    ax.legend(loc="upper right")
    ax.set_xlabel("$F_1$")
    ax.set_ylabel("$F_2$")
    fig.show()
    fig.savefig("mod_zdt_2d_illustration_f.pdf", format="pdf")


def main():
    draw_rule()
    draw_efficient_fronts()


if __name__ == "__main__":
    main()