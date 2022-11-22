import numpy as np


def evaluate(X,
             problem_args=None,
             constr=False):
    # evaluation function for gear train design problem
    """
    :param X: population of solutions
    :param problem_args: the arguments needed to define the problem
    :param constr: whether constraints are needed
    :return: evaluation metrics
    """
    # all the four variables must be inverger values
    x1 = np.round(X[:, 0])
    x2 = np.round(X[:, 1])
    x3 = np.round(X[:, 2])
    x4 = np.round(X[:, 3])

    # First original objective function
    f_0 = np.abs(6.931 - ((x3 / x1) * (x4 / x2)))
    # Second original objective function (the maximum value among the four variables)
    l = [x1, x2, x3, x4]
    f_1 = np.max(l, axis=0)

    g_0 = 0.5 - (f_0 / 6.931)
    G = np.where(g_0 < 0, -g_0, 0)

    F = np.column_stack((f_0, f_1))

    if constr:
        return F, G
    else:
        return F
