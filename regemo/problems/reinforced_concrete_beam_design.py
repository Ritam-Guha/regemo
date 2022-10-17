import numpy as np


def evaluate(X,
             problem_args,
             constr=False):
    # evaluation function for reinforced concrete beam design problem
    """
    :param X: population of solutions
    :param problem_args: the arguments needed to define the problem
    :param constr: whether constraints are needed
    :return: evaluation metrics
    """
    feasible_vals = np.array(
        [0.20, 0.31, 0.40, 0.44, 0.60, 0.62, 0.79, 0.80, 0.88, 0.93, 1.0, 1.20, 1.24, 1.32, 1.40, 1.55, 1.58, 1.60,
         1.76, 1.80, 1.86, 2.0, 2.17, 2.20, 2.37, 2.40, 2.48, 2.60, 2.64, 2.79, 2.80, 3.0, 3.08, 3, 10, 3.16, 3.41,
         3.52, 3.60, 3.72, 3.95, 3.96, 4.0, 4.03, 4.20, 4.34, 4.40, 4.65, 4.74, 4.80, 4.84, 5.0, 5.28, 5.40, 5.53, 5.72,
         6.0, 6.16, 6.32, 6.60, 7.11, 7.20, 7.80, 7.90, 8.0, 8.40, 8.69, 9.0, 9.48, 10.27, 11.0, 11.06, 11.85, 12.0,
         13.0, 14.0, 15.0])

    # Reference: getNearestValue_sample2.py (https://gist.github.com/icchi-h/1d0bb1c52ebfdd31f14b3e811328390a)
    idx = []
    for i in range(X.shape[0]):
        idx.append(np.abs(feasible_vals - X[i, 0]).argmin())
    x1 = feasible_vals[idx]
    x2 = X[:, 1]
    x3 = X[:, 2]

    # First original objective function
    f_0 = (29.4 * x1) + (0.6 * x2 * x3)

    # Original constraint functions
    g_0 = (x1 * x3) - 7.735 * ((x1 * x1) / x2) - 180.0
    g_1 = 4.0 - (x3 / x2)
    G = np.column_stack((g_0, g_1))
    G = np.where(G < 0, -G, 0)
    f_1 = np.sum(G, axis=1)
    F = np.column_stack((f_0, f_1))
    G = None

    if constr:
        return F, G
    else:
        return F
