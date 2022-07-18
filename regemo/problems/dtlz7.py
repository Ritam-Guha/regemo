from pymoo.problems.many.dtlz import DTLZ7

def evaluate(X,
             problem_args,
             constr=False):
    # evaluation function for DTLZ7 problem
    """
    :param X: population of solutions
    :param problem_args: the arguments needed to define the problem
    :param constr: whether constraints are needed
    :return: evaluation metrics
    """
    problem_class = DTLZ7(n_var=problem_args["dim"], n_obj=problem_args["n_obj"])
    out = {}

    problem_class._evaluate(X, out)
    F = out["F"]
    G = None
    if "G" in list(out.keys()):
        G = out["G"]

    if constr:
        return F, G
    else:
        return F