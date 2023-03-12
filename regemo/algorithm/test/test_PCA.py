from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.problems.multi import BNH
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter

import pickle
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np


def run():
    problem = BNH()

    algorithm = NSGA2(pop_size=100)

    res = minimize(problem,
                   algorithm,
                   ('n_gen', 200),
                   seed=1,
                   verbose=False)

    plot = Scatter()
    plot.add(problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
    plot.add(res.F, facecolor="none", edgecolor="red")
    plot.show()

    res_dict = {"X": res.X, "F": res.F}
    pickle.dump(res_dict, open("results.pickle", "wb"))


def pca_test(n_comps=1):
    # get the results and fit a PCA
    res = pickle.load(open("results.pickle", "rb"))
    X = res["X"][:, :2]
    pca = PCA(n_components=n_comps)
    pca.fit(X)

    plt.scatter(X[:, 0], X[:, 1], edgecolors="black")

    # calculate the slopes of the principal component
    x = X[:, 0]
    y = X[:, 1]
    x = np.sort(x)
    for i in range(n_comps):
        component_slope = pca.components_[i, 1]/pca.components_[i, 0]
        y_ = component_slope * x + min(y)
        idx_1 = np.argmin(x)
        idx_2 = np.argmax(x)
        plt.arrow(x[idx_1], y_[idx_1], x[idx_2], y_[idx_2], head_width=0.3, head_length=0.2, color='black')

    p_vals = pca.transform(X)
    # p_vals =
    plt.scatter(p_vals[:, 0], p_vals[:, 1], c="r", edgecolors="black")

    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.grid(alpha=0.5)
    plt.show()


def main():
    # run()
    pca_test(n_comps=1)


if __name__ == "__main__":
    main()