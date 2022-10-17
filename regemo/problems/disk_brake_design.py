import numpy as np


def evaluate(X,
             problem_args,
             constr=False):
    # evaluation function for disk brake design problem
    """
    :param X: population of solutions
    :param problem_args: the arguments needed to define the problem
    :param constr: whether constraints are needed
    :return: evaluation metrics
    """
    x1 = X[:, 0]
    x2 = X[:, 1]
    x3 = X[:, 2]
    x4 = X[:, 3]

    # First original objective function
    f_0 = 4.9 * 1e-5 * (x2 * x2 - x1 * x1) * (x4 - 1.0)
    # Second original objective function
    f_1 = ((9.82 * 1e6) * (x2 * x2 - x1 * x1)) / (x3 * x4 * (x2 * x2 * x2 - x1 * x1 * x1))

    # Reformulated objective functions
    g_0 = (x2 - x1) - 20.0
    g_1 = 0.4 - (x3 / (3.14 * (x2 * x2 - x1 * x1)))
    g_2 = 1.0 - (2.22 * 1e-3 * x3 * (x2 * x2 * x2 - x1 * x1 * x1)) / np.power((x2 * x2 - x1 * x1), 2)
    g_3 = (2.66 * 1e-2 * x3 * x4 * (x2 * x2 * x2 - x1 * x1 * x1)) / (x2 * x2 - x1 * x1) - 900.0
    G = np.column_stack((g_0, g_1, g_2, g_3))
    G = np.where(G < 0, -G, 0)

    F = np.column_stack((f_0, f_1))

    if constr:
        return F, G
    else:
        return F
