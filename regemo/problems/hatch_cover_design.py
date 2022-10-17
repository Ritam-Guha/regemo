import numpy as np


def evaluate(X,
             problem_args,
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

    # First original objective function
    f_0 = x1 + (120 * x2)

    E = 700000
    sigma_b_max = 700
    tau_max = 450
    delta_max = 1.5
    sigma_k = (E * x1 * x1) / 100
    sigma_b = 4500 / (x1 * x2)
    tau = 1800 / x2
    delta = (56.2 * 10000) / (E * x1 * x2 * x2)

    g_0 = 1 - (sigma_b / sigma_b_max)
    g_1 = 1 - (tau / tau_max)
    g_2 = 1 - (delta / delta_max)
    g_3 = 1 - (sigma_b / sigma_k)
    G = np.column_stack((g_0, g_1, g_2, g_3))
    G = np.where(G < 0, -G, 0)
    f_1 = np.sum(G, axis=1)

    F = np.column_stack((f_0, f_1))
    G = None

    if constr:
        return F, G
    else:
        return F
