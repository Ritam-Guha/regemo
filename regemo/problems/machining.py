import numpy as np


def evaluate(X,
             problem_args=None,
             constr=False):
    # evaluation function for hatch cover design problem
    """
    :param X: population of solutions
    :param problem_args: the arguments needed to define the problem
    :param constr: whether constraints are needed
    :return: evaluation metrics
    """
    x1 = X[:, 0]
    x2 = X[:, 1]
    x3 = X[:, 2]

    # objective functions
    f1 = -7.49 + (0.44 * x1) - (1.16 * x2) + (0.63 * x3)
    f2 = -4.31 + (0.92 * x1) - (0.16 * x2) + (0.43 * x3)
    f3 = 21.9 - (1.94 * x1) - (0.3 * x2) - (1.04 * x3)
    f4 = -11.331 + x1 + x2 + x3

    # constraints
    g1 = (-0.44 * x1) + (1.16 * x2) - (0.61 * x3) + 3.17
    g2 = (-0.92 * x1) + (0.16 * x2) - (0.43 * x3) + 8.04
    g3 = (-1.94 * x1) + (0.3 * x2) + (1.04 * x3) - 18.5
    F = np.column_stack((f1, f2, f3, f4))
    G = np.column_stack((g1, g2, g3))

    if constr:
        return F, G
    else:
        return F
