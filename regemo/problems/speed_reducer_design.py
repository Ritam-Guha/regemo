import numpy as np


def evaluate(X,
             problem_args=None,
             constr=False):
    # evaluation function for speed reducer design problem
    """
    :param X: population of solutions
    :param problem_args: the arguments needed to define the problem
    :param constr: whether constraints are needed
    :return: evaluation metrics
    """
    x1 = X[:, 0]
    x2 = X[:, 1]
    x3 = np.round(X[:, 2])
    x4 = X[:, 3]
    x5 = X[:, 4]
    x6 = X[:, 5]
    x7 = X[:, 6]

    # First original objective function (weight)
    f_0 = 0.7854 * x1 * (x2 * x2) * (((10.0 * x3 * x3) / 3.0) + (14.933 * x3) - 43.0934) - 1.508 * x1 * (
                x6 * x6 + x7 * x7) + 7.477 * (x6 * x6 * x6 + x7 * x7 * x7) + 0.7854 * (x4 * x6 * x6 + x5 * x7 * x7)

    # Second original objective function (stress)
    tmpVar = np.power((745.0 * x4) / (x2 * x3), 2.0) + 1.69 * 1e7
    f_1 = np.sqrt(tmpVar) / (0.1 * x6 * x6 * x6)

    # Constraint functions
    g_0 = -(1.0 / (x1 * x2 * x2 * x3)) + 1.0 / 27.0
    g_1 = -(1.0 / (x1 * x2 * x2 * x3 * x3)) + 1.0 / 397.5
    g_2 = -(x4 * x4 * x4) / (x2 * x3 * x6 * x6 * x6 * x6) + 1.0 / 1.93
    g_3 = -(x5 * x5 * x5) / (x2 * x3 * x7 * x7 * x7 * x7) + 1.0 / 1.93
    g_4 = -(x2 * x3) + 40.0
    g_5 = -(x1 / x2) + 12.0
    g_6 = -5.0 + (x1 / x2)
    g_7 = -1.9 + x4 - 1.5 * x6
    g_8 = -1.9 + x5 - 1.1 * x7
    g_9 = -f_1 + 1300.0
    tmpVar = np.power((745.0 * x5) / (x2 * x3), 2.0) + 1.575 * 1e8
    g_10 = -np.sqrt(tmpVar) / (0.1 * x7 * x7 * x7) + 1100.0
    G = np.column_stack((g_0, g_1, g_2, g_3, g_4, g_5, g_6, g_7, g_8, g_9, g_10))
    G = np.where(G < 0, -G, 0)

    F = np.column_stack((f_0, f_1))

    if constr:
        return F, G
    else:
        return F
