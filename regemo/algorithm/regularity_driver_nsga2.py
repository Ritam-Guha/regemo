import copy

import pandas as pd

from regemo.algorithm.regularity_search import Regularity_Search
import regemo.config as config
from regemo.problems.get_problem import problems as problem_set
from regemo.utils.path_utils import create_dir

from pymoo.core.problem import ElementwiseProblem
from pymoo.core.variable import Binary, Choice,  Integer, Real
from pymoo.algorithms.moo.nsga2 import RankAndCrowdingSurvival
from pymoo.core.mixed import MixedVariableGA
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from pymoo.core.callback import Callback
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.core.duplicate import DuplicateElimination

import pickle
import argparse
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import time


def get_unique_indices(F):
    view = F.view(np.dtype((np.void, F.dtype.itemsize * F.shape[1])))
    unique_rows, index = np.unique(view, return_index=True)
    return index


class RegEMOUpperLevelSearchProblem(ElementwiseProblem):
    def __init__(self, **kwargs):
        self.problem_config = kwargs["problem_config"]
        self.algorithm_config = kwargs["algorithm_config"]
        self.algorithm_config["NSGA_settings"]["n_eval"] = 5000

        vars = {
                "num_non_fixed_independent_vars": Integer(bounds=(1, min(10, self.problem_config["dim"]-1))),
                "non_fixed_regularity_degree": Integer(bounds=(1, 10)),
                "delta": Real(bounds=(1e-2, 0.5)),
                "n_rand_bins": Integer(bounds=(5, 10)),
                "precision": Integer(bounds=(2, 5))
        }

        super().__init__(vars=vars, n_obj=2, n_ieq_constr=0, **kwargs)

    def _evaluate(self, X, out, *args, **kwargs):
        # create a search object
        regularity_search = Regularity_Search(problem_args=self.problem_config,
                                              seed=config.seed,
                                              NSGA_settings=self.algorithm_config["NSGA_settings"],
                                              precision=X["precision"],
                                              n_rand_bins=X["n_rand_bins"],
                                              delta=X["delta"],
                                              non_fixed_regularity_degree=X["non_fixed_regularity_degree"],
                                              num_non_fixed_independent_vars=X["num_non_fixed_independent_vars"],
                                              save=False,
                                              verbose=False)

        # run the search object
        f1, f2 = regularity_search.run()

        out["F"] = [f1, f2]


class MyCallback(Callback):

    def __init__(self) -> None:
        super().__init__()

    def notify(self, algorithm):
        F = algorithm.pop.get("F")
        fronts = NonDominatedSorting().do(F)
        F = F[fronts[0], :]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(F[:, 0], F[:, 1], s=40, edgecolor="black")
        ax.set_xlabel("complexity", fontsize=12)
        ax.set_ylabel("$\Delta$HV (in %)", fontsize=12)
        ax.set_title(f"Gen: {algorithm.n_gen}", fontsize=14)
        ax.grid(alpha=0.3)
        fig.show()
        # print(f"Gen {algorithm.n_gen}")


def convert_pf_regularity_to_pd(param_comb):
    # convert the final PF regularities to a pandas dataframe
    df_dict = {}
    # first create dictionary with keys from the execution lists having empty lists
    for key in param_comb[0].keys():
        df_dict[key] = []

    # populate the empty lists with the param values
    for param in param_comb:
        for key in df_dict.keys():
            df_dict[key].append(param[key])

    pf_df = pd.DataFrame.from_dict(df_dict)

    return pf_df


def run_regularity_driver(problem_name):
    # collect arguments for the problem
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem_name", default="c2dtlz2", help="Name of the problem")
    parser.add_argument("--pop_size", default=20, type=int, help="Name of the problem")
    parser.add_argument("--n_eval", default=200, type=int, help="Name of the problem")
    args = parser.parse_args()
    problem_name = args.problem_name
    n_eval = args.n_eval
    pop_size = args.pop_size
    root_dir = "results/upper_level_search"
    create_dir(root_dir)
    start_time = time.time()

    if problem_name != "all":
        # if you want to run it on a specific problem
        problems = [problem_name]
    else:
        # if you want to run it on all the problems in regemo suite
        problems = problem_set

    # for the specified problems, run the regularity driver
    for problem_name in problems:
        algorithm_config_storage_dir = f"{config.BASE_PATH}/{config.algorithm_config_path}"
        problem_config_storage_dir = f"{config.BASE_PATH}/{config.problem_config_path}"
        use_existing_config = True
        algorithm_config, problem_config = {}, {}
        cur_dir = f"{root_dir}/{problem_name}"
        create_dir(cur_dir, delete=True)

        if use_existing_config:
            if not os.path.exists(f"{problem_config_storage_dir}/{problem_name}.pickle"):
                print("[Error!] Problem Configuration file not found...")
                sys.exit(1)
            if not os.path.exists(f"{algorithm_config_storage_dir}/{problem_name}.pickle"):
                print("[Error!] Algorithm Configuration file not found...")
                sys.exit(1)
            else:
                problem_config = pickle.load(open(f"{problem_config_storage_dir}/{problem_name}.pickle", "rb"))
                algorithm_config = pickle.load(open(f"{algorithm_config_storage_dir}/{problem_name}.pickle", "rb"))

        problem_config["problem_name"] = problem_name

        # define EMO algorithm
        problem = RegEMOUpperLevelSearchProblem(problem_config=problem_config,
                                                algorithm_config=algorithm_config)
        algorithm = MixedVariableGA(pop_size=pop_size, survival=RankAndCrowdingSurvival())
        res = minimize(problem,
                       algorithm,
                       ('n_eval', n_eval),
                       seed=config.seed,
                       callback=MyCallback(),
                       verbose=True,
                       save_history=True)

        # save all the points
        hist_f = []
        for algo in res.history:
            feas = np.where(algo.opt.get("feasible"))[0]
            hist_f.append(algo.opt.get("F")[feas])

        pickle.dump(hist_f, open(f"{config.BASE_PATH}/{cur_dir}/history.pickle", "wb"))

        # get the pareto front
        F, X = res.F, res.X
        front = NonDominatedSorting().do(F, only_non_dominated_front=True)
        F = F[front, :]
        X = X[front]

        # remove the duplicated entries
        unique_indices = get_unique_indices(F)
        F = F[unique_indices, :]
        X = X[unique_indices]

        sorted_idx = np.argsort(F[:, 0])
        F = F[sorted_idx, :]
        X = X[sorted_idx]
        pf = {"F": F,
              "X": X}
        pickle.dump(pf, open(f"{config.BASE_PATH}/{cur_dir}/upper_level_pareto_front.pickle", "wb"))

        # plot the Pareto front
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(F[:, 0], F[:, 1], s=40, edgecolor="black")
        ax.set_xlabel("complexity", fontsize=12)
        ax.set_ylabel("$\Delta$HV (in %)", fontsize=12)
        ax.grid(alpha=0.3)
        fig.savefig(f"{config.BASE_PATH}/{cur_dir}/upper_level_pareto_front.pdf", format="pdf", bbox_inches="tight")
        fig.show()

        # generate the configurations
        param_comb = copy.deepcopy(X)
        for i, x in enumerate(X):
            param_comb[i]["ID"] = i+1
            param_comb[i]["complexity"] = F[i, 0]
            param_comb[i]["hv_dif_%"] = F[i, 1]

            cur_dir = f"{root_dir}/{problem_name}/param_comb_{i+1}"
            create_dir(cur_dir, delete=True)

            # create a search object
            algorithm_config["NSGA_settings"]["n_eval"] = 5000
            regularity_search = Regularity_Search(problem_args=problem_config,
                                                  seed=config.seed,
                                                  NSGA_settings=algorithm_config["NSGA_settings"],
                                                  precision=x["precision"],
                                                  n_rand_bins=x["n_rand_bins"],
                                                  delta=x["delta"],
                                                  non_fixed_regularity_degree=x["non_fixed_regularity_degree"],
                                                  num_non_fixed_independent_vars=x["num_non_fixed_independent_vars"],
                                                  save=True,
                                                  result_storage=cur_dir,
                                                  verbose=False)
            regularity_search.run()

        # save the interactive pf config
        pf_df = convert_pf_regularity_to_pd(param_comb)
        fig = px.scatter(pf_df, x="complexity", y="hv_dif_%", hover_data=list(param_comb[0].keys()), log_x=False, size_max=60)
        fig.update_traces(textposition='top center', marker_size=10)
        fig.update_layout(
            height=800,
            title_text='Final Pareto Front Configuration for ' + problem_name
        )
        fig.show()
        fig.write_html(f"{config.BASE_PATH}/{root_dir}/{problem_name}/upper_level_pareto_front.html")
        end_time = time.time()
        print(f"Time required: {end_time - start_time}")


if __name__ == "__main__":
    run_regularity_driver(problem_name="BNH")