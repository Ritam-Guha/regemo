import numpy as np


def evaluate(X,
             problem_args=None,
             constr=False):
    # evaluation function for 2d  problem
    """
    :param X: population of solutions
    :param problem_args: the arguments needed to define the problem
    :param constr: whether constraints are needed
    :return: evaluation metrics
    """

    power = (X[:, 0] ** 2 + X[:, 1] ** 2 + X[:, 2] ** 2 - 1) ** 2 + (X[:, 3] ** 2) + ((X[:, 4] - 0.2) ** 2) + 0.5
    f1 = X[:, 0]
    f2 = X[:, 1]
    f3 = 1 - np.power(X[:, 0], power) - np.power(X[:, 1], power)
    F = np.column_stack((f1, f2, f3))

    if constr:
        return F, None

    else:
        return F