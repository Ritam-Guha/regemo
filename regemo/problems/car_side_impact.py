import numpy as np


def evaluate(X,
             problem_args=None,
             constr=False):
    # evaluation function for car side impact problem
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
    x5 = X[:, 4]
    x6 = X[:, 5]
    x7 = X[:, 6]

    # First original objective function
    f_0 = 1.98 + 4.9 * x1 + 6.67 * x2 + 6.98 * x3 + 4.01 * x4 + 1.78 * x5 + 0.00001 * x6 + 2.73 * x7
    # Second original objective function
    f_1 = 4.72 - 0.5 * x4 - 0.19 * x2 * x3
    # Third original objective function
    Vmbp = 10.58 - 0.674 * x1 * x2 - 0.67275 * x2
    Vfd = 16.45 - 0.489 * x3 * x7 - 0.843 * x5 * x6
    f_2 = 0.5 * (Vmbp + Vfd)

    # Constraint functions
    g_0 = 1 - (1.16 - 0.3717 * x2 * x4 - 0.0092928 * x3)
    g_1 = 0.32 - (0.261 - 0.0159 * x1 * x2 - 0.06486 * x1 - 0.019 * x2 * x7 + 0.0144 * x3 * x5 + 0.0154464 * x6)
    g_2 = 0.32 - (
                0.214 + 0.00817 * x5 - 0.045195 * x1 - 0.0135168 * x1 + 0.03099 * x2 * x6 - 0.018 * x2 * x7 + 0.007176 * x3 + 0.023232 * x3 - 0.00364 * x5 * x6 - 0.018 * x2 * x2)
    g_3 = 0.32 - (0.74 - 0.61 * x2 - 0.031296 * x3 - 0.031872 * x7 + 0.227 * x2 * x2)
    g_4 = 32 - (28.98 + 3.818 * x3 - 4.2 * x1 * x2 + 1.27296 * x6 - 2.68065 * x7)
    g_5 = 32 - (33.86 + 2.95 * x3 - 5.057 * x1 * x2 - 3.795 * x2 - 3.4431 * x7 + 1.45728)
    g_6 = 32 - (46.36 - 9.9 * x2 - 4.4505 * x1)
    g_7 = 4 - f_1
    g_8 = 9.9 - Vmbp
    g_9 = 15.7 - Vfd
    G = np.column_stack((g_0, g_1, g_2, g_3, g_4, g_5, g_6, g_7, g_8, g_9))
    G = np.where(G < 0, -G, 0)

    F = np.column_stack((f_0, f_1, f_2))

    if constr:
        return F, G
    else:
        return F
