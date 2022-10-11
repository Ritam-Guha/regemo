import numpy as np


def evaluate(X,
             problem_args,
             constr=False):
    # evaluation function for Welded Beam Design problem
    """
    :param X: population of solutions
    :param problem_args: the arguments needed to define the problem
    :param constr: whether constraints are needed
    :return: evaluation metrics
    """
    num_solutions, num_features = X.shape
    # def f1
    f1 = (1.10471 * (X[:, 0] ** 2) * X[:, 1]) + (0.04811 * X[:, 2] * X[:, 3] * (14.0 + X[:, 1]))

    # def f2
    f2 = 2.1952 / ((X[:, 2] ** 3) * X[:, 3])

    # def constraints
    r1 = 6000 / (np.sqrt(2) * X[:, 0] * X[:, 1])
    r2 = (6000 * (14 + (0.5 * X[:, 1])) * np.sqrt(0.25 * ((X[:, 1] ** 2) + ((X[:, 0] + X[:, 2]) ** 2)))) / (
            2 * (0.707 * X[:, 0] * X[:, 1] * ((X[:, 1] ** 2) / 12) + 0.25 * ((X[:, 0] + X[:, 2]) ** 2)))
    r = np.sqrt((r1 ** 2) + (r2 ** 2) + ((X[:, 1] * r1 * r2) / np.sqrt(0.25 * ((X[:, 1] ** 2) + ((X[:,
                                                                                                  0] + X[:,
                                                                                                       2]) **
                                                                                                 2)))))
    sigma = 504000 / ((X[:, 2] ** 2) * X[:, 3])
    pc = 64746.022 * (1 - (0.0282346 * X[:, 2])) * (X[:, 2] * (X[:, 3] ** 3))

    g1 = r - 13600
    g2 = sigma - 30000
    g3 = X[:, 0] - X[:, 3]
    g4 = 6000 - pc
    # g5 = 2.5094 - f1

    F = np.column_stack([f1, f2])
    G = np.column_stack([g1, g2, g3, g4])

    if constr:
        # return np.column_stack([f1, f2]), np.column_stack([g1, g2, g3, g4, g5])
        return F, G
    else:
        return F