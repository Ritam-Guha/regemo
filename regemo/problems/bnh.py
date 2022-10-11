from pymoo.problems.multi.bnh import BNH


def evaluate(X,
             problem_args,
             constr=False):

    # evaluation function for BNH problem
    """
    :param X: population of solutions
    :param problem_args: the arguments needed to define the problem
    :param constr: whether constraints are needed
    :return: evaluation metrics
    """
    problem_class = BNH()
    out = {}
    problem_class._evaluate(X, out)
    F = out["F"]

    if "G" in list(out.keys()):
        G = out["G"]

    # g3 = 72-F[:, 0]
    # G = np.column_stack([G, g3])
    if constr:
        return F, G
    else:
        return F