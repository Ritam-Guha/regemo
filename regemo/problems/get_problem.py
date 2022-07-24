from regemo.problems import bnh
from regemo.problems import c2dtlz2
from regemo.problems import crashworthiness
from regemo.problems import dtlz2
from regemo.problems import dtlz5
from regemo.problems import dtlz7
from regemo.problems import mod_zdt
from regemo.problems import osy
from regemo.problems import scalable_truss
from regemo.problems import srn
from regemo.problems import two_member_truss
from regemo.problems import welded_beam_design

from pymoo.core.problem import Problem
import numpy as np
import copy

problems = ["bnh", "c2dtlz2", "crashworthiness", "dtlz2", "dtlz5", "dtlz7", "mod_zdt", "osy", "srn",
            "two_member_truss", "welded_beam_design", "scalable_truss"]

evaluation_mapper = {
    "bnh": bnh.evaluate,
    "c2dtlz2": c2dtlz2.evaluate,
    "crashworthiness": crashworthiness.evaluate,
    "dtlz2": dtlz2.evaluate,
    "dtlz5": dtlz5.evaluate,
    "dtlz7": dtlz7.evaluate,
    "mod_zdt": mod_zdt.evaluate,
    "osy": osy.evaluate,
    "scalable_truss": scalable_truss.evaluate,
    "srn": srn.evaluate,
    "two_member_truss": two_member_truss.evaluate,
    "welded_beam_design": welded_beam_design.evaluate
}

def get_problem(problem_name,
                problem_args,
                class_required=True):

    """
    :param problem_name: name of the problem to be solved
    :param problem_args: arguments for the problem
    :param class_required: if class is required as an outcome
    :return: evaluation function/class for the problem
    """

    assert(problem_name.lower() in list(evaluation_mapper.keys()))

    # get the evaluation function for the corresponding problem
    evaluate = evaluation_mapper[problem_name.lower()]

    if class_required:
        class Test(Problem):
            def __init__(self):
                if len(problem_args['lb']) < problem_args['dim']:
                    problem_args['lb'] = problem_args['lb'] * problem_args['dim']
                    problem_args['ub'] = problem_args['ub'] * problem_args['dim']

                super().__init__(n_var=problem_args['dim'],
                                 n_obj=problem_args['n_obj'],
                                 n_constr=problem_args['n_constr'],
                                 xl=problem_args['lb'],
                                 xu=problem_args['ub'])

            def _evaluate(self, 
                          X, 
                          out, 
                          *args, 
                          **kwargs):

                if "regularity_enforcement" in problem_args.keys() and problem_args["regularity_enforcement"]:
                    # if the algorithm wants to enforce the regularity
                    # mapping to original dimension
                    new_dim = problem_args["dim"] + len(problem_args["non_rand_vars"]) + len(problem_args[
                                                                                                 "rand_dependent_vars"])
                    new_X = np.zeros((X.shape[0], new_dim))
                    
                    # placing the fixed values in the new population
                    new_X[:, problem_args["rand_independent_vars"]] = X[:, problem_args["rand_variable_mapper"][
                                                                               "rand_independent_vars"]]
                    new_X[:, problem_args["rand_complete_vars"]] = X[:, problem_args["rand_variable_mapper"][
                                                                            "rand_complete_vars"]]
                    new_X = problem_args["regularity_enforcement_process"](new_X)
                    
                    # setting the new problem arguments
                    new_problem_args = copy.deepcopy(problem_args)
                    new_problem_args["dim"] = new_dim

                else:
                    new_X = copy.deepcopy(X)
                    new_problem_args = copy.deepcopy(problem_args)

                # evaluate the population
                F, G = evaluate(new_X, new_problem_args, constr=True)

                # add the bound constraints
                if "regularity_enforcement" in new_problem_args.keys() and new_problem_args["regularity_enforcement"]:
                    for i, idx in enumerate(new_problem_args["rand_dependent_vars"]):
                        if G is None:  # first time G may be None
                            G = new_X[:, idx] - new_problem_args["rand_dependent_ub"][i]
                        else:
                            G = np.column_stack((G, new_X[:, idx] - new_problem_args["rand_dependent_ub"][i]))
                        G = np.column_stack((G, new_problem_args["rand_dependent_lb"][i] - new_X[:, idx]))

                out["F"], out["G"] = F, G

        return Test()

    else:
        return evaluate

