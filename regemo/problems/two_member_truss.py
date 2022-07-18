import numpy as np

def evaluate(X,
             problem_args,
             constr=False):
    # evaluation function for Two Member Truss problem
    """
    :param X: population of solutions
    :param problem_args: the arguments needed to define the problem
    :param constr: whether constraints are needed
    :return: evaluation metrics
    """
    num_solutions, num_features = X.shape

    # def f1
    f1 = X[:, 0] * np.sqrt(16 + (X[:, 2] ** 2)) + X[:, 1] * np.sqrt(1 + (X[:, 2] ** 2))

    # def f2
    sigma_ac = (20 * np.sqrt(16 + (X[:, 2] ** 2))) / (X[:, 2] * X[:, 0])
    sigma_bc = (80 * np.sqrt(1 + (X[:, 2] ** 2))) / (X[:, 2] * X[:, 1])

    sigma = np.column_stack([sigma_ac, sigma_bc])
    f2 = np.max(sigma, axis=1)

    g1 = f2 - np.power(10, 5)
    # g2 = f1 - 0.04472

    F = np.column_stack([f1, f2])
    G = np.column_stack([g1])

    if constr:
        # return np.column_stack([f1, f2]), np.column_stack([g1, g2])
        return F, G
    else:
        return F