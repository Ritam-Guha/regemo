from regemo.problems.get_problem import problems as problem_set
from regemo.problems.get_problem import get_problem
from regemo.algorithm.regularity_finder import Regularity_Finder
from regemo.utils.path_utils import create_dir
import regemo.config as config
from regemo.algorithm.nsga3 import NSGA3

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import HyperplaneNormalization
from pymoo.algorithms.moo.nsga3 import associate_to_niches
from pymoo.visualization.scatter import Scatter
from pymoo.factory import get_termination
from pymoo.optimize import minimize
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.visualization.radviz import Radviz
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.operators.crossover.sbx import SBX

import inspect
import argparse
import copy
import os
import sys
from pymoo.visualization.pcp import PCP
import plotly.express as px
import plotly.io as pio

import matplotlib.pyplot as plt
import numpy as np
import pickle
import subprocess
import pandas as pd
# from pdflatex import PDFLaTeX
import warnings

from yellowbrick.features import RadViz as RadVizYellow

warnings.filterwarnings("ignore")


# plt.rcParams.update({'font.size': 10})

color_mapping_orig_F = {
    0: "blue",
    1: "orange",
    2: "purple"
}

color_mapping_reg_F = {
    0: "brown",
    1: "olive",
    2: "gray"
}


def get_default_args(func):
    """
    :param func: function definition
    :return: get the default arguments of the function
    """
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }


class Regularity_Search:
    # upper level class for regularity enforcement
    def __init__(self,
                 problem_args,
                 num_non_fixed_independent_vars=1,
                 non_fixed_regularity_degree=1,
                 delta=0.05,
                 precision=2,
                 n_rand_bins=20,
                 NSGA_settings=None,
                 save=True,
                 result_storage=None,
                 verbose=True,
                 seed=0,
                 n_processors=1):
        """
        :param problem_args: parameters for the problem
        :param delta: threshold for identifying fixed variables
        :param precision: precisions of the floating point numbers
        :param n_rand_bins: number of bins used to identify random variables
        :param NSGA_settings: parametric settings for the algorithm
        :param save: whether to save the resulting images
        :param result_storage: storage place for the results
        :param verbose: whether to print the console log
        :param seed: seed for the run
        """
        super().__init__()

        # set problem-specific information
        self.regularity_objs = None
        self.problem_name = problem_args["name"]
        self.problem_args = problem_args
        self.seed = seed
        self.num_non_fixed_independent_vars = num_non_fixed_independent_vars
        self.non_fixed_regularity_degree = non_fixed_regularity_degree
        self.NSGA_settings = NSGA_settings
        self.n_rand_bins = n_rand_bins
        self.delta = delta

        # get the problem utilities
        self.problem = get_problem(self.problem_name, problem_args)
        self.evaluate = get_problem(self.problem_name, problem_args, class_required=False)

        # initialize the regularity algorithm parameters
        self.regularity_obj = None
        self.orig_X = None
        self.orig_F = None
        self.regular_X = None
        self.regular_F = None

        self.combined_X = None
        self.combined_F = None
        self.result_storage = result_storage
        self.precision = precision
        self.save = save
        self.verbose = verbose
        self.print = self.verboseprint()
        self.visualization_angle = self.problem_args["visualization_angle"]
        self.ideal_point = None
        self.nadir_point = None
        self.n_processors = n_processors

        # final objectives
        self.hv = None
        self.final_metrics = {
            "complexity": 0,
            "hv_dif_%": 0,
            "igd_lus": 0
        }

        np.random.seed(self.seed)

    def run(self):
        # the main running function
        initial_pop_storage = f"{config.BASE_PATH}/results/initial_populations/" \
                              f"initial_population_{self.problem_name}.pickle"

        if os.path.exists(initial_pop_storage):
            res = pickle.load(open(initial_pop_storage, "rb"))

        else:
            res = self.run_NSGA(self.problem, self.NSGA_settings)
            # store the initial population
            initial_population = {"X": res.X, "F": res.F}
            create_dir(os.path.dirname(initial_pop_storage.replace(config.BASE_PATH, "")))
            with open(initial_pop_storage, "wb") as file_handle:
                pickle.dump(initial_population, file_handle)
            res = {"X": res.X, "F": res.F}

        # ideal and nadir point estimation
        if self.problem_args["n_obj"] > 2:
            self.edge_point_estimation(self.NSGA_settings["ref_dirs"], res["F"])

        if self.save:
            # plot the figure after nds
            plot = Scatter(labels="F", legend=False, angle=self.visualization_angle)
            plot = plot.add(res["F"], color="blue", label="Original PO Front", alpha=0.2, s=60)
            plot.save(f"{config.BASE_PATH}/{self.result_storage}/initial_efficient_front.pdf", format="pdf")

        self.orig_X = res["X"]
        self.orig_F = res["F"]

        # new NSGA settings
        new_NSGA_settings = copy.deepcopy(self.NSGA_settings)
        new_NSGA_settings["ideal_point"] = self.ideal_point
        new_NSGA_settings["nadir_point"] = self.nadir_point
        new_NSGA_settings["pop_size"] = self.orig_X.shape[0]

        if self.problem_args["n_obj"] > 2:
            cur_ref_dirs = self.check_closest_ref_dirs(self.NSGA_settings["ref_dirs"], res["F"])
            new_NSGA_settings["ref_dirs"] = cur_ref_dirs

        # use the regularity enforcement object to extract the regularity
        regularity_enforcement = Regularity_Finder(X=self.orig_X,
                                                   F=self.orig_F,
                                                   problem_args=self.problem_args,
                                                   num_non_fixed_independent_vars=self.num_non_fixed_independent_vars,
                                                   non_fixed_regularity_degree=self.non_fixed_regularity_degree,
                                                   precision=self.precision,
                                                   NSGA_settings=new_NSGA_settings,
                                                   seed=self.seed,
                                                   save=self.save,
                                                   result_storage=self.result_storage,
                                                   verbose=self.verbose,
                                                   n_rand_bins=self.n_rand_bins,
                                                   delta=self.delta,
                                                   n_processors=self.n_processors)

        regularity_enforcement.run()    # run the regularity enforcement
        self.final_metrics["complexity"] += regularity_enforcement.regularity.calc_process_complexity()
        self.final_metrics["igd_plus"] = regularity_enforcement.final_metrics["igd_plus"]
        self.final_metrics["hv_dif_%"] = regularity_enforcement.final_metrics["hv_dif_%"]

        # storage file for every PF
        if self.save:
            text_storage = f"{config.BASE_PATH}/{self.result_storage}/regularity_pf.txt"
            tex_storage = f"{config.BASE_PATH}/{self.result_storage}/regularity_pf.tex"
            tex_storage_long = f"{config.BASE_PATH}/{self.result_storage}/regularity_pf_long.tex"

            # store the regularity in text and tex files
            regularity_enforcement.regularity.display(self.orig_X,
                                                      self.problem_args["lb"],
                                                      self.problem_args["ub"],
                                                      save_file=text_storage)

            regularity_enforcement.regularity.display_tex(self.orig_X,
                                                          self.problem_args["lb"],
                                                          self.problem_args["ub"],
                                                          save_file=tex_storage)

            regularity_enforcement.regularity.display_tex_long(self.orig_X,
                                                               self.problem_args["lb"],
                                                               self.problem_args["ub"],
                                                               save_file=tex_storage_long)

        # subprocess.run(['pdflatex', '-output-directory',
        #                 f'{config.BASE_PATH}/{self.result_storage}/',
        #                 tex_storage_long])

        # self.print("Final Metrics")
        # self.print(f"IGD+: {regularity_enforcement.final_metrics['igd_plus']}")
        # self.print(f"HV_dif_%: {regularity_enforcement.final_metrics['hv_dif_%']}")
        self.print("\n======================================\n")

        # plot the regular front
        if self.save:
            # get the X and F after regularity enforcement
            self.regular_X = regularity_enforcement.X
            self.regular_F = regularity_enforcement.F

            # save the objects
            self.regularity_obj = regularity_enforcement

            if self.regular_F is not None:
                plot = Scatter(labels="F", legend=False, angle=self.visualization_angle)
                plot = plot.add(self.regular_F, color="red", marker="*", s=60, alpha=0.6,
                                label="Regular Efficient Front")
                plot.save(f"{config.BASE_PATH}/{self.result_storage}/regular_efficient_front.pdf", format="pdf")
                if self.verbose and plot:
                    plot.show()
                plt.close()

        # plot the original and regular front
            plot = Scatter(labels="F", legend=True, angle=self.visualization_angle, tight_layout=True, fontsize=5)
            plot = plot.add(self.orig_F, color="blue", marker="o", s=60, alpha=0.2, label="Original PO Front")
            if self.regular_F is not None:
                plot = plot.add(self.regular_F, color="red", marker="*", s=50, alpha=0.6, label="Regular Front")

                front_df_reg = pd.DataFrame()
                for i in range(self.regular_F.shape[1]):
                    front_df_reg[f"F_{i}"] = self.regular_F[:, i]
                front_df_reg["type"] = "regular"
                front_df_orig = pd.DataFrame()
                for i in range(self.orig_F.shape[1]):
                    front_df_orig[f"F_{i}"] = self.orig_F[:, i]
                front_df_orig["type"] = "original"
                front_df = pd.concat((front_df_reg, front_df_orig))

                fig = None
                if self.orig_F.shape[1] == 2:
                    fig = px.scatter(front_df, x="F_0", y="F_1", color="type")
                elif self.orig_F.shape[1] == 3:
                    fig = px.scatter_3d(front_df, x="F_0", y="F_1", z="F_2", color="type")
                else:
                    # [NOTE] plot high-dimensional points
                    plot = Radviz(legend=(True, {'loc': "upper left", 'bbox_to_anchor': (-0.1, 1.08, 0, 0)}))
                    plot.set_axis_style(color="black", alpha=1.0)
                    plot.add(self.orig_F, color="blue", s=40, label="Original PO Front")
                    plot.add(self.regular_F, color="red", s=40, marker="*", label="Regular Front")
                    plot.show()

                    original_df = pd.DataFrame(self.orig_F, columns=[f"$f_{i+1}$" for i in range(self.problem_args[
                                                                                                     "n_obj"])])
                    regular_df = pd.DataFrame(self.regular_F, columns=[f"$f_{i + 1}$" for i in range(self.problem_args[
                                                                                                       "n_obj"])])
                    X = pd.concat((original_df, regular_df))
                    y = np.zeros(original_df.shape[0] + regular_df.shape[0])
                    y[original_df.shape[0]:] = 1
                    classes = ["Original PO Front", "Regular Front"]
                    visualizer = RadVizYellow(classes=classes, colors=["blue", "red"], grid=False)
                    visualizer.fit(X, y)  # Fit the data to the visualizer
                    visualizer.transform(X)  # Transform the data
                    plt.savefig(f"{config.BASE_PATH}/{self.result_storage}/final_efficient_fronts.png", dpi=200)
                    visualizer.show()  # Finalize and render the figure


                if fig:
                    pio.write_html(fig, file=f'{config.BASE_PATH}/{self.result_storage}/final_efficient_fronts.html')
                plot.save(f"{config.BASE_PATH}/{self.result_storage}/final_efficient_fronts.pdf", format="pdf", dpi=200)

                if self.verbose:
                    plot.show()

            plt.close()

            with open(f"{config.BASE_PATH}/{self.result_storage}/initial_population.pickle",
                      "wb") as file_handle:
                pickle.dump(res, file_handle)

            # store the final population
            final_population = {"X": self.regular_X, "F": self.regular_F}
            with open(f"{config.BASE_PATH}/{self.result_storage}/final_regular_population.pickle",
                      "wb") as file_handle:
                pickle.dump(final_population, file_handle)

            # get a pcp plot of the final population
            if self.regular_X is not None:
                plot = PCP(labels="X")
                plot.normalize_each_axis = False
                plot.set_axis_style(color="grey", alpha=0.5)
                plot.add(self.regular_X)
                plot.save(f"{config.BASE_PATH}/{self.result_storage}/PCP_final_population.pdf", format="pdf")

                if self.verbose:
                    plot.show()

            plt.close()

            with open(f"{config.BASE_PATH}/{self.result_storage}/final_metrics.txt", "w") as f:
                f.write(f"HV_dif_%: {self.final_metrics['hv_dif_%']}\n")
                f.write(f"complexity: {self.final_metrics['complexity']}\n")
                f.write(f"IGD+: {self.final_metrics['igd_plus']}")

        self.print(f"hv_diff_%: {self.final_metrics['hv_dif_%']}")
        self.print(f"IGD+: {self.final_metrics['igd_plus']}")
        self.print(f"complexity: {self.final_metrics['complexity']}")

        return self.final_metrics["complexity"], self.final_metrics["hv_dif_%"]

    def run_NSGA(self, problem, NSGA_settings):
        # run the NSGA2 over the problem
        if self.problem_args["n_obj"] == 2:
            # if we are dealing with a 2-objective problem, use NSGA2
            self.print("Running NSGA2..")
            algorithm = NSGA2(pop_size=NSGA_settings["pop_size"],
                              n_offsprings=NSGA_settings["n_offsprings"],
                              sampling=FloatRandomSampling(),
                              crossover=SBX(prob=NSGA_settings["sbx_prob"],
                                            eta=NSGA_settings["sbx_eta"]),
                              mutation=PolynomialMutation(eta=NSGA_settings["mut_eta"]),
                              seed=self.seed,
                              eliminate_duplicate=True,
                              verbose=self.verbose)

        elif self.problem_args["n_obj"] > 2:
            # for working with many-objective problems, use NSGA3
            algorithm = NSGA3(pop_size=NSGA_settings["pop_size"],
                              n_offsprings=NSGA_settings["n_offsprings"],
                              sampling=FloatRandomSampling(),
                              crossover=SBX(prob=NSGA_settings["sbx_prob"],
                                            eta=NSGA_settings["sbx_eta"]),
                              mutation=PolynomialMutation(eta=NSGA_settings["mut_eta"]),
                              ref_dirs=NSGA_settings["ref_dirs"],
                              seed=self.seed,
                              eliminate_duplicate=True,
                              verbose=self.verbose)

        else:
            print("[INFO] Not suitable for less than 2 objectives...")
            sys.exit(1)

        # define the termination criterion
        termination = get_termination("n_eval", NSGA_settings["n_eval"])

        # start the minimization process
        res = minimize(problem,
                       algorithm,
                       termination,
                       seed=self.seed,
                       verbose=self.verbose,
                       save_history=True)

        hist_f = []
        for algo in res.history:
            feas = np.where(algo.opt.get("feasible"))[0]
            hist_f.append(algo.opt.get("F")[feas])

        if self.save:
            pickle.dump(hist_f, open(f"{config.BASE_PATH}/{self.result_storage}/regular_convergence_history.pickle",
                                     "wb"))
        return res

    def edge_point_estimation(self,
                              ref_dirs,
                              F):
        # calculate the fronts of the population
        fronts, rank = NonDominatedSorting().do(F, return_rank=True)
        non_dominated, last_front = fronts[0], fronts[-1]

        # update the hyperplane based boundary estimation
        hyp_norm = HyperplaneNormalization(ref_dirs.shape[1])
        hyp_norm.update(F, nds=non_dominated)

        # update the ideal point and nadir point
        self.ideal_point, self.nadir_point = hyp_norm.ideal_point, hyp_norm.nadir_point

    def check_closest_ref_dirs(self,
                               ref_dirs,
                               F):
        # get the closest reference directions for every point in F
        # associate individuals to niches
        niche_of_individuals, dist_to_niche, dist_matrix = associate_to_niches(F,
                                                                               ref_dirs,
                                                                               self.ideal_point,
                                                                               self.nadir_point)
        closest_ref_dir_index = np.argmin(dist_matrix, axis=1)

        closest_ref_dirs = ref_dirs[closest_ref_dir_index, :]
        return closest_ref_dirs

    @staticmethod
    def _normalize(x, lb, ub):
        # function to normalize x between 0 and 1
        new_x = copy.deepcopy(x)

        if new_x.ndim == 1:
            new_x = np.array([new_x])

        for i in range(new_x.shape[1]):
            new_x[:, i] = (new_x[:, i] - lb[i]) / (ub[i] - lb[i])

        if new_x.shape[0] == 1:
            # converting a single array back to a 1D array
            new_x = new_x[0, :]

        return new_x

    def verboseprint(self):
        if self.verbose:
            def mod_print(*args, end="\n"):
                # Print each argument separately so caller doesn't need to
                # stuff everything to be printed into a single string
                for arg in args:
                    print(arg, end=end),
                print
        else:
            def mod_print(*args, end="\n"):
                pass

        return mod_print


def main(problem_name="water",
         **kwargs):
    # collect arguments for the problem
    seed = config.seed
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem_name", default=problem_name, help="Name of the problem")
    args = parser.parse_args()
    problem_name = args.problem_name

    if problem_name != "all":
        # if you want to run it on a specific problem
        problems = [problem_name]
    else:
        # if you want to run it on all the problems in regemo suite
        problems = problem_set

    for problem_name in problems:
        res_storage_dir = f"results/{problem_name}"
        algorithm_config_storage_dir = config.algorithm_config_path
        problem_config_storage_dir = config.problem_config_path

        # create the dirs for storing images and config files
        create_dir(res_storage_dir, delete=True)

        if not os.path.exists(f"{config.BASE_PATH}/{problem_config_storage_dir}/{problem_name}.pickle"):
            print("[Error!] Problem Configuration file not found...")
            sys.exit(1)
        if not os.path.exists(f"{config.BASE_PATH}/{algorithm_config_storage_dir}/{problem_name}.pickle"):
            print("[Error!] Algorithm Configuration file not found...")
            sys.exit(1)
        else:
            problem_config = pickle.load(
                open(f"{config.BASE_PATH}/{problem_config_storage_dir}/{problem_name}.pickle", "rb"))
            algorithm_config = pickle.load(
                open(f"{config.BASE_PATH}/{algorithm_config_storage_dir}/{problem_name}.pickle", "rb"))

        print(problem_config)
        print(algorithm_config)

        # create a search object
        regularity_search = Regularity_Search(problem_args=problem_config,
                                              seed=seed,
                                              NSGA_settings=algorithm_config["NSGA_settings"],
                                              precision=10,
                                              n_rand_bins=5,
                                              delta=0.5,
                                              non_fixed_regularity_degree=2,
                                              num_non_fixed_independent_vars=1,
                                              save=True,
                                              result_storage=f"{res_storage_dir}",
                                              verbose=True,
                                              n_processors=4)

        # run the search object
        regularity_search.run()


if __name__ == "__main__":
    main()
