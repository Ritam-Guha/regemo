import numpy as np
from sklearn.linear_model import LinearRegression
import copy


def form_problem_pattern(x,
                         dim):

    ### Used for building the modified ZDT problem ###

    """
    :param x: population of solutions
    :param dim: dimension of the problem
    :return: a new defined problem
    """

    # define pattern of the problem
    new_x = np.zeros(dim)

    for idx_list in x:
        (indices, start_point, choice) = idx_list

        if choice == 'const':
            new_x[indices] = [start_point] * len(indices)

        elif choice == 'linear':
            new_x[indices] = start_point + 0.05 * np.arange(len(indices))

        elif choice == 'piecewise':
            new_x[indices[0]] = start_point

            for i in range(1, len(indices)):
                if i < int(len(indices) / 2):
                    new_x[indices[i]] = new_x[indices[i-1]] + 0.05

                else:
                    new_x[indices[i]] = new_x[indices[i - 1]] - 0.05

    return list(new_x)

def fit_curve(X,
              clusters,
              degree=1):

    """
    :param X: population of solutions
    :param clusters: cluster in the population
    :param degree: the degree of fitting a curve through a cluster
    :return: new population after fitting
    """

    # fit a linear regression curve taking care of the degree
    new_X = copy.deepcopy(X)

    for pattern_indices in clusters:
        if len(pattern_indices) > 1:
            # x_1 represents the variables in degree 1
            x_1 = np.arange(len(pattern_indices)).reshape(-1, 1)
            x = x_1.copy()

            for j in range(1, degree):
                # x_degree represents the variables with degrees higher than 1
                x_degree = x_1 ** (j + 1)
                x = np.append(x, x_degree, axis=1)

            # y is the corresponding target values
            y = X[pattern_indices].reshape(-1, 1)

            # fit the regression
            reg = LinearRegression().fit(x, y)

            # get the fitted predictions to the corresponding places
            predicted_pattern = reg.predict(x)
            new_X[pattern_indices] = predicted_pattern[:, 0]

    return new_X

def verboseprint(verbose):
    if verbose:
        def mod_print(*args, end="\n"):
            # Print each argument separately so caller doesn't need to
            # stuff everything to be printed into a single string
            for arg in args:
                print(arg, end=end),
            print
    else:
        def mod_print(*args, end="\n"):
            pass

    return mod_print


if __name__ == '__main__':
    pass