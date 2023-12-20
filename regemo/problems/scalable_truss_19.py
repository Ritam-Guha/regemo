from regemo.problems.scalable_truss_impl.truss.truss_problem_general import TrussProblemGeneral
import multiprocessing

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
                                        n_cores=4,
                                        repair_inequality=False,
                                        repair_power=False,
                                        symmetry=())
    out = {}

    try:
        problem_class._evaluate(X, out)
    except:
        print(X)

    F = out["F"]
    if "G" in list(out.keys()):
        G = out["G"]

    if constr:
        return F, G
    else:
        return F