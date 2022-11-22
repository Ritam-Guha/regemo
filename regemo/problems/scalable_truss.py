from regemo.problems.scalable_truss_impl.truss.truss_problem_general import TrussProblemGeneral


def evaluate(X,
             problem_args=None,
             constr=False):
    # evaluation function for Scalable Truss problem
    """
    :param X: population of solutions
    :param problem_args: the arguments needed to define the problem
    :param constr: whether constraints are needed
    :return: evaluation metrics
    """
    problem_class = TrussProblemGeneral(n_shape_var=19,
                                        n_cores=1,
                                        repair_inequality=False,
                                        repair_power=False,
                                        repair_interval=10,
                                        symmetry=())
    out = {}

    problem_class._evaluate(X, out)
    F = out["F"]
    if "G" in list(out.keys()):
        G = out["G"]

    if constr:
        return F, G
    else:
        return F