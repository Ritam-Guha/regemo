import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from sklearn.linear_model import LinearRegression
from pymoo.visualization.scatter import Scatter
from regemo.problems.get_problem import zdt_mod_3d
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


def fit_linreg(x1, x2, x3):
    y = x1
    X = np.column_stack((x2, x3, x2**2, x3**2, x2*x3))
    fit = LinearRegression().fit(X, y)
    return fit


def draw_sphere():
    n_points = 10000
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(projection='3d')
    np.random.seed(0)
    x2 = np.outer(np.linspace(0, 1, n_points), np.ones(n_points))
    x3 = x2.copy().T
    x1 = np.sqrt(1 - x2 ** 2 - x3 ** 2)

    ax.view_init(10, -58)
    ax.plot_surface(x1, x2, x3, color="blue", alpha=0.8, label=f"original rule: $x_1^2 + x_2^2 + x_3^2 = 1$")
    ax.set_xlabel("$x_1$", fontsize=15)
    ax.set_ylabel("$x_2$", fontsize=15)
    ax.set_zlabel("$x_3$", fontsize=15)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)

    np.random.seed(0)
    x1 = np.ravel(x1)
    x2 = np.ravel(x2)
    x3 = np.ravel(x3)

    not_nan = np.where(~np.isnan(x1))[0]

    fit = fit_linreg(x1[not_nan], x2[not_nan], x3[not_nan])
    x1_reg_non_lin = fit.predict(np.column_stack((x2, x3, x2 ** 2, x3 ** 2, x2*x3)))
    reg_non_lin_valid = np.where(np.logical_and(0 <= x1_reg_non_lin, x1_reg_non_lin <= 1))
    x2 = x2.reshape(n_points, n_points)
    x3 = x3.reshape(n_points, n_points)
    x1_reg_non_lin = x1_reg_non_lin.reshape(n_points, n_points)
    # ax.plot_surface(x1_reg_non_lin, x2, x3, color="green", alpha=0.6,
    #         label=f"regular non-linear rule: $x_1 = {np.round(fit.coef_[0], 2)} x_2 + {np.round(fit.coef_[1], 2)} x_3 + {np.round(fit.coef_[2], 2)} x_2^2 + {np.round(fit.coef_[3], 2)} x_3^2 + {np.round(fit.intercept_, 2)} x_2 x_3$")
    return fig, ax, fit


def draw_rule():
    n_points = 1000
    fig, ax, fit = draw_sphere()
    np.random.seed(0)
    x2 = np.outer(np.linspace(0, 1, n_points), np.ones(n_points))
    x1 = x2.copy().T
    x3 = - (0.9 * x2) - (0.6 * x1) + 1.2
    ax.plot_surface(x1, x2, x3, color="red", alpha=0.3, label="original rule: $x_1 + 0.5x_2 + x_3 = 1.23$")
    fake2Dline_1 = mpl.lines.Line2D([0], [0], linestyle="none", c='b', marker='o')
    # fake2Dline_2 = mpl.lines.Line2D([0], [0], linestyle="none", c='g', marker='o')
    fake2Dline_3 = mpl.lines.Line2D([3], [3], linestyle="none", c='r', marker='o')
    ax.legend([fake2Dline_1,
               # fake2Dline_2,
               fake2Dline_3],
              ['original rule: $x_1^2 + x_2^2 + x_3^2 = 1$',
                # f"regular non-linear rule: $x_1 = {np.round(fit.coef_[0], 2)} x_2 + {np.round(fit.coef_[1], 2)} x_3$ \n${np.round(fit.coef_[2], 2)} x_2^2 {np.round(fit.coef_[3], 2)} x_3^2 + {np.round(fit.intercept_, 2)} x_2 x_3 + {np.round(fit.intercept_, 2)}$",
               "regular linear rule: $0.6x_1 + 0.9x_2 + x_3 = -1.2$"], numpoints=1, fontsize=15)
    fig.savefig("mod_zdt_3d_illustration_x.png")


def draw_efficient_fronts():
    n_points = 1000
    np.random.seed(0)
    x2 = np.random.rand(n_points)
    x3 = np.random.rand(n_points)
    x1_orig = np.sqrt(1 - x2 ** 2 - x3 ** 2)
    x1_reg_lin = 1.23 - x3 - 0.5 * x2

    not_nan = np.where(~np.isnan(x1_orig))[0]

    fit = fit_linreg(x1_orig[not_nan], x2[not_nan], x3[not_nan])
    x1_reg_non_lin = fit.predict(np.column_stack((x2, x3, x2 ** 2, x3 ** 2, x2*x3)))
    x4 = np.ones(len(x1_orig)) * 0
    x5 = np.ones(len(x1_orig)) * 0.2
    F_orig = zdt_mod_3d.evaluate(np.column_stack((x1_orig, x2, x3, x4, x5)))
    F_reg_lin = zdt_mod_3d.evaluate(np.column_stack((x1_reg_lin, x2, x3, x4, x5)))
    F_reg_non_lin = zdt_mod_3d.evaluate(np.column_stack((x1_reg_non_lin, x2, x3, x4, x5)))

    reg_lin_valid = np.where(np.logical_and(0 <= x1_reg_lin, x1_reg_lin <= 1))
    reg_non_lin_valid = np.where(np.logical_and(0 <= x1_reg_non_lin, x1_reg_non_lin <= 1))

    F_reg_lin = F_reg_lin[reg_lin_valid]
    F_reg_non_lin = F_reg_non_lin[reg_non_lin_valid]

    # plot the figure after nds
    plot = Scatter(labels="F", legend=True, angle=(29, -60))
    plot = plot.add(F_orig, color="blue", label="original efficient front", alpha=0.8, s=60)
    fronts = NonDominatedSorting().do(F_reg_lin)
    plot = plot.add(F_reg_lin[fronts[0]], color="red", label="regular linear efficient front", alpha=0.2, s=60)
    # fronts = NonDominatedSorting().do(F_reg_non_lin)
    # plot = plot.add(F_reg_non_lin[fronts[0]], color="green", label="regular non-linear efficient front", alpha=0.6, s=60)
    plot.show()
    plot.save("mod_zdt_3d_illustration_f.png")


def main():
    draw_rule()
    draw_efficient_fronts()


if __name__ == "__main__":
    main()