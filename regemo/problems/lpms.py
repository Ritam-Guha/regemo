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
    x_1 = X[:, 0]
    x_2 = X[:, 1]

    def u1(x1, x2):
        return 3 * ((1-x1)**2) * np.exp(-x1**2 - (x2+1)**2)

    def u2(x1, x2):
        return -10 * (x1/(4 - x1**3 - x2**5)) * np.exp(- x1**2 * x2**2)

    def u3(x1, x2):
        return (1/3) * np.exp(-(x1+1)**2 - x2**2)

    def f(x1, x2):
        return -u1(x1, x2) - u2(x1, x2) - u3(x1, x2)

    # First original objective function
    f1 = f(x_1, x_2)
    f2 = f(x_1 - 1.2, x_2 - 1.5)
    f3 = f(x_1 + 0.3, x_2 - 3)
    f4 = f(x_1 - 1, x_2 + 0.5)
    f5 = f(x_1 - 0.5, x_2 - 1.7)

    F = np.column_stack((f1, f2, f3, f4, f5))
    G = None

    if constr:
        return F, G
    else:
        return F
