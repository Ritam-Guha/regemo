from regemo.problems.solid_rocket_design.rocket_propellant_design.rocket import RocketProblem
import numpy as np


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
    problem_class = RocketProblem(use_parallelization=True)
    out = {}

    problem_class._evaluate(X, out)
    F = out["F"]

    if "G" in list(out.keys()):
        G = out["G"]

    if constr:
        return F, G
    else:
        return F
