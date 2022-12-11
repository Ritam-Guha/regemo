import autograd.numpy as anp

from pymoo.core.problem import Problem
from pymoo.util.remote import Remote


class TNK(Problem):
    def __init__(self,
                 n_var=2,
                 n_obj=2,
                 n_constr=2):
        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=n_constr, type_var=anp.double)
        self.xl = anp.array([0, 1e-30])
        self.xu = anp.array([anp.pi, anp.pi])

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = x[:, 0]
        f2 = x[:, 1]
        g1 = -(anp.square(x[:, 0]) + anp.square(x[:, 1]) - 1.0 - 0.1 * anp.cos(16.0 * anp.arctan(x[:, 0] / x[:, 1])))
        g2 = 2 * (anp.square(x[:, 0] - 0.5) + anp.square(x[:, 1] - 0.5)) - 1

        out["F"] = anp.column_stack([f1, f2])
        out["G"] = anp.column_stack([g1, g2])

    def _calc_pareto_front(self, *args, **kwargs):
        return Remote.get_instance().load("pf", "tnk.pf")

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