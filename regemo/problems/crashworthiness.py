import numpy as np


def evaluate(X,
             problem_args=None,
             constr=False):
    # evaluation function for Crashworthiness problem
    """
    :param X: population of solutions
    :param problem_args: the arguments needed to define the problem
    :param constr: whether constraints are needed
    :return: evaluation metrics
    """

    Mass = 1640.2823 + 2.3573285 * X[:, 0] + 2.3220035 * X[:, 1] + 4.5688768 * X[:, 2] + 7.7213633 * X[:, 3] + 4.4559504 \
           * X[:, 4]

    Ain = 6.5856 + 1.15 * X[:, 0] - 1.0427 * X[:, 1] + 0.9738 * X[:, 2] + 0.8364 * X[:, 3] - 0.3695 * X[:, 0] * X[:, 3] \
          + 0.0861 * X[:, 0] * X[:, 4] + 0.3628 * X[:, 1] * X[:, 3] - 0.1106 * (X[:, 0] ** 2) - 0.3437 * (X[:, 2] ** 2) \
          + 0.1764 * (X[:, 3] ** 2)

    Intrusion = -0.0551 + 0.0181 * X[:, 0] + 0.1024 * X[:, 1] + 0.0421 * X[:, 2] - 0.0073 * X[:, 0] * X[:, 1] + 0.0204 * \
                X[:, 1] * X[:, 2] - 0.0118 * X[:, 1] * X[:, 3] - 0.0204 * X[:, 2] * X[:, 3] - 0.008 * X[:, 2] * X[:, 4] \
                - 0.0241 * (X[:, 1] ** 2) + 0.0109 * (X[:, 3] ** 2)

    f1, f2, f3 = Mass, Ain, Intrusion
    F = np.column_stack([f1, f2, f3])

    if constr:
        return F, None

    else:
        return F