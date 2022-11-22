import numpy as np


def evaluate(X,
             problem_args=None,
             constr=False):
    # evaluation function for Crashworthiness problem
    """
    :param X: population of solutions
    :param problem_args: the arguments needed to define the problem
    :param constr: whether constraints are needed
    :return: evaluation metrics
    """

    power = (X[:, 0] ** 2 + X[:, 1] ** 2 - 1) ** 2 + 0.25
    f1 = X[:, 0]
    f2 = 1 - np.power(X[:, 0], power)
    F = np.column_stack((f1, f2))

    if constr:
        return F, None

    else:
        return F