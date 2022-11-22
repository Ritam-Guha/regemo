import numpy as np


def evaluate(X,
             problem_args=None,
             constr=False):
    # evaluation function for pressure vessel design problem
    """
    :param X: population of solutions
    :param problem_args: the arguments needed to define the problem
    :param constr: whether constraints are needed
    :return: evaluation metrics
    """
    x1 = 0.0625 * np.int64(np.round(X[:, 0]))
    x2 = 0.0625 * np.int64(np.round(X[:, 1]))
    x3 = X[:, 2]
    x4 = X[:, 3]

    # First original objective function
    f_0 = (0.6224 * x1 * x3 * x4) + (1.7781 * x2 * x3 * x3) + (3.1661 * x1 * x1 * x4) + (19.84 * x1 * x1 * x3)

    # Original constraint functions
    g_0 = x1 - (0.0193 * x3)
    g_1 = x2 - (0.00954 * x3)
    g_2 = (np.pi * x3 * x3 * x4) + ((4.0 / 3.0) * (np.pi * x3 * x3 * x3)) - 1296000
    G = np.column_stack((g_0, g_1, g_2))
    G = np.where(G < 0, -G, 0)
    f_1 = np.sum(G, axis=1)

    F = np.column_stack((f_0, f_1))
    G = None

    if constr:
        return F, G
    else:
        return F
