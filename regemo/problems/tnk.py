from pymoo.problems.multi.tnk import TNK


def evaluate(X,
             problem_args,
             constr=False):
    # evaluation function for TNK problem
    """
    :param X: population of solutions
    :param problem_args: the arguments needed to define the problem
    :param constr: whether constraints are needed
    :return: evaluation metrics
    """
    problem_class = TNK(n_var=problem_args["dim"], n_obj=problem_args["n_obj"])
    out = {}
    problem_class._evaluate(X, out)
    F = out["F"]
    G = out["G"]

    if constr:
        return F, G
    else:
        return F