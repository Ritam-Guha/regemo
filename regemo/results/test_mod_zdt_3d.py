import regemo.config as config
from sklearn.linear_model import LinearRegression
from regemo.problems.get_problem import get_problem

import pickle
import matplotlib.pyplot as plt
import numpy as np


def get_fitting(problem_name):
    problem_evaluate = get_problem(problem_name,
                                   problem_args=None,
                                   class_required=False)
    init_pop = pickle.load(open(f"{config.BASE_PATH}/results/hierarchical_search/{problem_name}/"
                                f"initial_population_{problem_name}.pickle", "rb"))

    init_pop_x = init_pop["X"]
    init_pop_f = init_pop["F"]

    X_1 = (-1 * init_pop_x[:, 2]) + (-0.4 * init_pop_x[:, 1]) + 1.23
    reg_pop_x = np.column_stack((X_1, init_pop_x[:, 1], init_pop_x[:, 2]))

    fig = plt.figure(figsize=(10, 7))
    ax = plt.axes(projection="3d")
    ax.scatter3D(init_pop_x[:, 0], init_pop_x[:, 1], init_pop_x[:, 2], c="r", label="Original Pareto Front")
    ax.scatter3D(reg_pop_x[:, 0], reg_pop_x[:, 1], reg_pop_x[:, 2], c="b", label="Regular Pareto Front")
    ax.set_xlabel("x_1")
    ax.set_ylabel("x_2")
    ax.set_ylabel("x_3")
    ax.legend(loc="lower right")
    fig.savefig(f"{config.BASE_PATH}/results/hierarchical_search/{problem_name}/design_space_plot.jpg")
    plt.show()

    # fig, ax = plt.subplots()
    # ax.scatter(init_pop_f[:, 0], init_pop_f[:, 1], c="r", label="Original Efficient Front")
    # ax.scatter(reg_pop_f[:, 0], reg_pop_f[:, 1], c="b", label="Regressed Efficient Front")
    # ax.set_xlabel("f_1")
    # ax.set_ylabel("f_2")
    # ax.annotate(f"x_2 = {np.round(reg.coef_[0][0], 2)} * x_1 + {np.round(reg.intercept_[0], 2)}", xy=(0.1, 0.8))
    # ax.legend(loc="lower right")
    # fig.savefig(f"{config.BASE_PATH}/results/hierarchical_search/{problem_name}/efficient_space_plot.jpg")
    # plt.show()


def main():
    problem_name = "zdt_mod_3d"
    get_fitting(problem_name)


if __name__ == "__main__":
    main()
