from pymoo.problems.multi.osy import OSY


def evaluate(X,
             problem_args,
             constr=False):
    # evaluation function for OSY problem
    """
    :param X: population of solutions
    :param problem_args: the arguments needed to define the problem
    :param constr: whether constraints are needed
    :return: evaluation metrics
    """
    problem_class = OSY()
    out = {}

    problem_class._evaluate(X, out)
    F = out["F"]
    if "G" in list(out.keys()):
        G = out["G"]

    if constr:
        return F, G
    else:
        return F