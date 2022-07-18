from pymoo.problems.many.cdtlz import C2DTLZ2

def evaluate(X,
             problem_args,
             constr=False):
    # evaluation function for C2DTLZ2 problem
    """
    :param X: population of solutions
    :param problem_args: the arguments needed to define the problem
    :param constr: whether constraints are needed
    :return: evaluation metrics
    """
    problem_class = C2DTLZ2(n_var=problem_args["dim"], n_obj=problem_args["n_obj"])
    out = {}
    problem_class._evaluate(X, out)
    F = out["F"]
    G = out["G"]

    if constr:
        return F, G
    else:
        return F