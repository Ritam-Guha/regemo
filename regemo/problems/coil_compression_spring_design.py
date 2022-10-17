import numpy as np


def evaluate(X,
             problem_args,
             constr=False):
    # evaluation function for coil compression spring design problem
    """
    :param X: population of solutions
    :param problem_args: the arguments needed to define the problem
    :param constr: whether constraints are needed
    :return: evaluation metrics
    """
    feasible_vals = np.array(
        [0.009, 0.0095, 0.0104, 0.0118, 0.0128, 0.0132, 0.014, 0.015, 0.0162, 0.0173, 0.018, 0.02, 0.023, 0.025, 0.028,
         0.032, 0.035, 0.041, 0.047, 0.054, 0.063, 0.072, 0.08, 0.092, 0.105, 0.12, 0.135, 0.148, 0.162, 0.177, 0.192,
         0.207, 0.225, 0.244, 0.263, 0.283, 0.307, 0.331, 0.362, 0.394, 0.4375, 0.5])
    x1 = np.round(X[:, 0])
    x2 = X[:, 1]
    # Reference: getNearestValue_sample2.py (https://gist.github.com/icchi-h/1d0bb1c52ebfdd31f14b3e811328390a)
    # Reference: getNearestValue_sample2.py (https://gist.github.com/icchi-h/1d0bb1c52ebfdd31f14b3e811328390a)
    idx = []
    for i in range(X.shape[0]):
        idx.append(np.abs(feasible_vals - X[i, 2]).argmin())
    x3 = feasible_vals[idx]

    # first original objective function
    f_0 = (np.pi * np.pi * x2 * x3 * x3 * (x1 + 2)) / 4.0

    # constraint functions
    Cf = ((4.0 * (x2 / x3) - 1) / (4.0 * (x2 / x3) - 4)) + (0.615 * x3 / x2)
    Fmax = 1000.0
    S = 189000.0
    G = 11.5 * 1e+6
    K = (G * x3 * x3 * x3 * x3) / (8 * x1 * x2 * x2 * x2)
    lmax = 14.0
    lf = (Fmax / K) + 1.05 * (x1 + 2) * x3
    dmin = 0.2
    Dmax = 3
    Fp = 300.0
    sigmaP = Fp / K
    sigmaPM = 6
    sigmaW = 1.25

    g_0 = -((8 * Cf * Fmax * x2) / (np.pi * x3 * x3 * x3)) + S
    g_1 = -lf + lmax
    g_2 = -3 + (x2 / x3)
    g_3 = -sigmaP + sigmaPM
    g_4 = -sigmaP - ((Fmax - Fp) / K) - 1.05 * (x1 + 2) * x3 + lf
    g_5 = sigmaW - ((Fmax - Fp) / K)
    G = np.column_stack((g_0, g_1, g_2, g_3, g_4, g_5))

    G = np.where(G < 0, -G, 0)
    f_1 = np.sum(G, axis=1)

    F = np.column_stack((f_0, f_1))
    G = None

    if constr:
        return F, G
    else:
        return F
