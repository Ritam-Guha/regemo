import numpy as np

def evaluate(X,
             problem_args=None,
             constr=False):
    # evaluation function for re_21 four bar truss design problem
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

    F_ = 10.0
    sigma = 10.0
    E = 2.0 * 1e5
    L = 200.0

    f_0 = L * ((2 * x1) + np.sqrt(2.0) * x2 + np.sqrt(x3) + x4)
    f_1 = ((F_ * L) / E) * ((2.0 / x1) + (2.0 * np.sqrt(2.0) / x2) - (2.0 * np.sqrt(2.0) / x3) + (2.0 / x4))
    F = np.column_stack((f_0, f_1))
    G = None

    if constr:
        return F, G
    else:
        return F
