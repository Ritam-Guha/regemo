import numpy as np


def evaluate(X,
             problem_args,
             constr=False):
    # evaluation function for rocket injector design problem
    """
    :param X: population of solutions
    :param problem_args: the arguments needed to define the problem
    :param constr: whether constraints are needed
    :return: evaluation metrics
    """
    # all the four variables must be inverger values
    xAlpha = X[:, 0]
    xHA = X[:, 1]
    xOA = X[:, 2]
    xOPTT = X[:, 3]

    # f1 (TF_max)
    f_0 = 0.692 + (0.477 * xAlpha) - (0.687 * xHA) - (0.080 * xOA) - (0.0650 * xOPTT) - (0.167 * xAlpha * xAlpha) - (
                0.0129 * xHA * xAlpha) + (0.0796 * xHA * xHA) - (0.0634 * xOA * xAlpha) - (0.0257 * xOA * xHA) + (
                       0.0877 * xOA * xOA) - (0.0521 * xOPTT * xAlpha) + (0.00156 * xOPTT * xHA) + (
                       0.00198 * xOPTT * xOA) + (0.0184 * xOPTT * xOPTT)
    # f2 (X_cc)
    f_1 = 0.153 - (0.322 * xAlpha) + (0.396 * xHA) + (0.424 * xOA) + (0.0226 * xOPTT) + (0.175 * xAlpha * xAlpha) + (
                0.0185 * xHA * xAlpha) - (0.0701 * xHA * xHA) - (0.251 * xOA * xAlpha) + (0.179 * xOA * xHA) + (
                       0.0150 * xOA * xOA) + (0.0134 * xOPTT * xAlpha) + (0.0296 * xOPTT * xHA) + (
                       0.0752 * xOPTT * xOA) + (0.0192 * xOPTT * xOPTT)
    # f3 (TT_max)
    f_2 = 0.370 - (0.205 * xAlpha) + (0.0307 * xHA) + (0.108 * xOA) + (1.019 * xOPTT) - (0.135 * xAlpha * xAlpha) + (
                0.0141 * xHA * xAlpha) + (0.0998 * xHA * xHA) + (0.208 * xOA * xAlpha) - (0.0301 * xOA * xHA) - (
                       0.226 * xOA * xOA) + (0.353 * xOPTT * xAlpha) - (0.0497 * xOPTT * xOA) - (
                       0.423 * xOPTT * xOPTT) + (0.202 * xHA * xAlpha * xAlpha) - (0.281 * xOA * xAlpha * xAlpha) - (
                       0.342 * xHA * xHA * xAlpha) - (0.245 * xHA * xHA * xOA) + (0.281 * xOA * xOA * xHA) - (
                       0.184 * xOPTT * xOPTT * xAlpha) - (0.281 * xHA * xAlpha * xOA)

    F = np.column_stack((f_0, f_1, f_2))
    G = None

    if constr:
        return F, G
    else:
        return F
