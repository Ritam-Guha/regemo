import numpy as np

def evaluate(X,
             problem_args,
             constr=False):
    # evaluation function for water problem
    """
    :param X: population of solutions
    :param problem_args: the arguments needed to define the problem
    :param constr: whether constraints are needed
    :return: evaluation metrics
    """

    x1, x2, x3 = X[:, 0], X[:, 1], X[:, 2]

    f1 = 106780.37 * (x2 + x3) + 61704.67
    f2 = 3000 * x1
    f3 = (305700 * 2289 * x2) / ((0.06 * 2289) ** 0.65)
    f4 = 250 * 2289 * np.exp(-39.75 * x2 + 9.9 * x3 + 2.74)
    f5 = 25 * (1.39 / (x1 * x2) + 4940 * x3 - 80)

    g1 = 0.00139 / (x1 * x2) + 4.94 * x3 - 0.08
    g1 = -(1 - g1) / 1

    g2 = 0.000306 / (x1 * x2) + 1.082 * x3 - 0.0986
    g2 = -(1 - g2) / 1

    g3 = 12.307 / (x1 * x2) + 49408.24 * x3 + 4051.02
    g3 = -(50000 - g3) / 50000

    g4 = 2.098 / (x1 * x2) + 8046.33 * x3 - 696.71
    g4 = -(16000 - g4) / 16000

    g5 = 2.138 / (x1 * x2) + 7883.39 * x3 - 705.04
    g5 = -(10000 - g5) / 10000

    g6 = 0.417 / (x1 * x2) + 1721.26 * x3 - 136.54
    g6 = -(2000 - g6) / 2000

    g7 = 0.164 / (x1 * x2) + 631.13 * x3 - 54.48
    g7 = -(550 - g7) / 550

    F = np.column_stack([f1, f2, f3, f4, f5])
    G = np.column_stack([g1, g2, g3, g4, g5, g6, g7])

    if constr:
        return F, G

    else:
        return F