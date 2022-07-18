import numpy as np

def evaluate(X,
             problem_args,
             constr=False):
    # evaluation function for modified ZDT problem
    """
    :param X: population of solutions
    :param problem_args: the arguments needed to define the problem
    :param constr: whether constraints are needed
    :return: evaluation metrics
    """
    num_solutions, num_features = X.shape
    # define f1
    f1 = X[:, 0]

    # define f2
    center_coef = problem_args['center_coef'][1:]
    center_offset = problem_args['center_offset'][1:]

    center_coef = np.array([i if i > 0 else -i for i in center_coef])
    centers = np.zeros((num_solutions, num_features - 1))

    for i in range(num_solutions):
        centers[i, :] = center_coef * X[i, 0] + center_offset

    sigma = np.sum(abs(X[:, 1:] - centers), axis=1)
    g = 1 + 9.0 / (problem_args['dim'] - 1) * sigma
    f2 = g * (1 - np.power((f1 / g), 0.5))

    F = np.column_stack([f1, f2])

    if constr:
        return F, None
    else:
        return F