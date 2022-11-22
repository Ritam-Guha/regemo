import numpy as np


def evaluate(X,
             problem_args=None,
             constr=False):
    # evaluation function for OSY problem
    """
    :param X: population of solutions
    :param problem_args: the arguments needed to define the problem
    :param constr: whether constraints are needed
    :return: evaluation metrics
    """
    out = {}

    f1 = 2 + np.power(X[:, 0] - 2, 2) + np.power(X[:, 1] - 1, 2)
    f2 = (9 * X[:, 0]) - np.power(X[:, 1] - 1, 2)

    g1 = np.power(X[:, 0], 2) + np.power(X[:, 1], 2) - 225
    g2 = X[:, 0] - (3 * X[:, 1]) + 10

    F = np.column_stack([f1, f2])
    G = np.column_stack([g1, g2])

    if constr:
        return F, G
    else:
        return F