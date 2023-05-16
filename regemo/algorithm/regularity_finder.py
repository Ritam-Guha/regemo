from regemo.problems.get_problem import get_problem
from regemo.utils.algo_utils import fit_curve, verboseprint
from regemo.algorithm.regularity import Regularity
import regemo.config as config
from regemo.algorithm.nsga3 import NSGA3

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.core.population import pop_from_array_or_individual
from pymoo.algorithms.moo.nsga2 import RankAndCrowdingSurvival
from pymoo.algorithms.moo.nsga3 import ReferenceDirectionSurvival
from pymoo.factory import get_sampling, get_crossover, get_mutation, get_termination, get_performance_indicator
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter

from sklearn.linear_model import Ridge as linreg
from sklearn.metrics import mean_squared_error as MSE

import copy
import sys
from tabulate import tabulate

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import binned_statistic
import pandas as pd
import pickle

plt.rcParams.update({'font.size': 15})


class Regularity_Finder:
    def __init__(self,
                 X,
                 F,
                 problem_args,
                 non_fixed_dependency_percent=0.9,
                 non_fixed_regularity_coef_factor=0.1,
                 non_fixed_regularity_degree=1,
                 delta=0.05,
                 precision=2,
                 n_rand_bins=20,
                 NSGA_settings=None,
                 save_img=True,
                 result_storage="/.",
                 verbose=True,
                 seed=0):
        """
        :param X: original PO solutions in X-space
        :param F: original PO solution in F-space
        :param problem_args: parameters to the problem
        :param non_fixed_dependency_percent: percentage of non-fixed variables to be treated as non-fixed dependent
        :param non_fixed_regularity_coef_factor: coefficients should be a multiple of this factor
        :param delta: threshold for identifying fixed variables
        :param non_fixed_regularity_degree: degree for non-fixed variables
        :param precision: precisions of the floating point numbers
        :param n_rand_bins: number of bins used to identify random variables
        :param NSGA_settings: parametric settings for the algorithm
        :param save_img: whether to save the resulting images
        :param result_storage: storage place for the results
        :param verbose: whether to print the console log
        :param seed: seed for the run
        """

        super().__init__()

        # set problem-specific information
        self.proxy_regular_F = None
        self.proxy_regular_X = None
        self.problem_name = problem_args["name"]
        self.problem_args = problem_args
        self.seed = seed
        self.n_rand_bins = n_rand_bins
        self.non_fixed_dependency_percent = non_fixed_dependency_percent
        self.non_fixed_regularity_coef_factor = non_fixed_regularity_coef_factor
        self.delta = delta
        self.non_fixed_regularity_degree = non_fixed_regularity_degree
        self.NSGA_settings = NSGA_settings

        # get the problem utilities
        self.problem = get_problem(self.problem_name, problem_args)
        self.evaluate = get_problem(self.problem_name, problem_args, class_required=False)

        # initialize the regularity algorithm parameters
        self.fixed_vars = []
        self.fixed_vals = None
        self.non_fixed_vars = None
        self.X = X
        self.F = F
        self.norm_F = None
        self.norm_F_lb = None
        self.norm_F_ub = None
        self.intermediate_X = None
        self.intermediate_F = None
        self.orig_X = None
        self.orig_F = None
        self.orig_hv = None
        self.regularity_hv = None
        self.norm_orig_F = None
        self.ub = list(np.array(problem_args["ub"], dtype=float))
        self.lb = list(np.array(problem_args["lb"], dtype=float))
        self.pareto_lb = None
        self.pareto_ub = None
        self.non_fixed_final_reg_coef_list = []
        self.non_fixed_dependent_vars = []
        self.non_fixed_independent_vars = []
        self.non_fixed_orphan_vars = []
        self.non_fixed_degree_list = []
        self.result_storage = result_storage
        self.igd_plus = None
        self.hv = None
        self.precision = precision
        self.save_img = save_img
        self.verbose = verbose
        self.print = verboseprint(self.verbose)
        self.regularity = None
        self.final_metrics = {}

        np.random.seed(self.seed)

        # validate the user inputs
        self._validate_inputs()

    def _validate_inputs(self):
        # function to validate user inputs
        default_NSGA_settings = {"pop_size": 100,
                                 "n_offsprings": 30,
                                 "sbx_prob": 0.8,
                                 "sbx_eta": 25,
                                 "mut_eta": 30,
                                 "n_eval": 40000}

        if self.NSGA_settings is None:
            # if NSGA_settings is not specified, use the default one
            self.NSGA_settings = default_NSGA_settings
        else:
            # if NSGA_settings is specified, but all the parameters are not properly
            # set the remaining parameters from the default setting
            missing_keys = list(np.setdiff1d(list(default_NSGA_settings.keys()), list(self.NSGA_settings.keys())))
            for i in missing_keys:
                self.NSGA_settings[i] = default_NSGA_settings[i]

    def run(self):
        # main code to run the regularity enforcing algorithm
        self.orig_X, self.orig_F = copy.deepcopy(self.X), copy.deepcopy(self.F)
        self.pareto_lb = np.round(np.min(self.orig_X, axis=0), self.precision)
        self.pareto_ub = np.round(np.max(self.orig_X, axis=0), self.precision)

        # set the performance indicators
        self.igd_plus = get_performance_indicator("igd+", self.orig_F)
        self.hv = get_performance_indicator("hv", ref_point=2 * np.ones(self.problem_args["n_obj"]))

        # set the initial metrics
        self.norm_F_lb = np.min(self.orig_F, axis=0)
        self.norm_F_ub = np.max(self.orig_F, axis=0)
        self.norm_orig_F = self._normalize(self.orig_F, self.norm_F_lb, self.norm_F_ub)
        self.orig_hv = self.hv.do(self.norm_orig_F)

        # dividing variables into fixed and non-fixed
        self.non_fixed_vars, self.fixed_vars = self.find_regularity_clusters()

        # regularity enforcement in non-fixed variables
        if len(self.non_fixed_vars) > 0:
            self.print("\n================================================")
            self.print("Searching for regularity inside non-fixed variables...")
            self.print("================================================")
            self.non_fixed_regularity()
            if self.non_fixed_dependent_vars:
                self.print("The regularity is that the slopes of the projection of the variables with respect to the "
                           "index variable are always multiples of " + str(self.non_fixed_regularity_coef_factor))

        if len(self.fixed_vars) > 0:
            # regularity enforcement in fixed variables
            self.print("================================================")
            self.print("Searching for regularity inside fixed variables...")
            self.print("================================================")
            self.fixed_regularity()
            self.print("The regularity is that all the population members are having the same values for the fixed "
                       "variables")
            # after the fixed values are found, save those values in fixed_vals
            self.fixed_vals = list(self.X[0, self.fixed_vars])

        else:
            self.print("[INFO] The process is stopping as there's no fixed variables")

        # save the regularity to a regularity object
        self.regularity = Regularity(dim=self.problem_args["dim"],
                                     lb=self.pareto_lb,
                                     ub=self.pareto_ub,
                                     fixed_vars=self.fixed_vars,
                                     fixed_vals=self.fixed_vals,
                                     non_fixed_vars=self.non_fixed_vars,
                                     non_fixed_dependent_vars=self.non_fixed_dependent_vars,
                                     non_fixed_independent_vars=self.non_fixed_independent_vars,
                                     non_fixed_orphan_vars=self.non_fixed_orphan_vars,
                                     non_fixed_final_reg_coef_list=self.non_fixed_final_reg_coef_list,
                                     non_fixed_degree_list=self.non_fixed_degree_list,
                                     problem_configs=self.problem_args,
                                     precision=self.precision)

        if self.verbose:
            self.regularity.display(self.orig_X)

        self.X = self.regularity.apply(self.X)
        self.F = self.evaluate(self.X, problem_args=self.problem_args)

        # plot the regular front
        plot = Scatter(labels="F", legend=True, angle=self.problem_args["visualization_angle"])
        plot = plot.add(self.orig_F, color="blue", marker="o", s=15, label="Original Efficient Front")
        plot = plot.add(self.F, color="red", marker="*", s=40, label="Regular Efficient Front")

        # plot.title = "Regular Efficient Front (Before Re-optimization)"

        if self.verbose:
            plot.show()

        if self.save_img:
            plot.save(
                f"{config.BASE_PATH}/{self.result_storage}/regular_efficient_front_pre_reopt.pdf", format="pdf")

        # Re-optimization
        if len(self.non_fixed_vars) > 0:
            self.print("\nRe-optimizing the population after regularity enforcement...")
            self.re_optimize()

            # save the regularity to a regularity object
            self.regularity = Regularity(dim=self.problem_args["dim"],
                                         lb=self.pareto_lb,
                                         ub=self.pareto_ub,
                                         fixed_vars=self.fixed_vars,
                                         fixed_vals=self.fixed_vals,
                                         non_fixed_vars=self.non_fixed_vars,
                                         non_fixed_dependent_vars=self.non_fixed_dependent_vars,
                                         non_fixed_independent_vars=self.non_fixed_independent_vars,
                                         non_fixed_orphan_vars=self.non_fixed_orphan_vars,
                                         non_fixed_final_reg_coef_list=self.non_fixed_final_reg_coef_list,
                                         non_fixed_degree_list=self.non_fixed_degree_list,
                                         problem_configs=self.problem_args,
                                         precision=self.precision)

            if self.X is not None:
                pickle.dump(
                    (len(self.non_fixed_independent_vars) + len(self.non_fixed_orphan_vars), self.problem_args["dim"]),
                    open(f"{config.BASE_PATH}/{self.result_storage}/non_fixed_var_count.pickle",
                         "wb"))

                # final metric calculation
                self.norm_F = self._normalize(self.F, self.norm_F_lb, self.norm_F_ub)
                if self.norm_F.ndim == 1:
                    self.norm_F = self.norm_F.reshape(1, -1)

                # save the final metrics
                self.regularity_hv = self.hv.do(self.norm_F)

                if self.regularity_hv == 0:
                    # when it converges to 1 point
                    self.final_metrics["hv_dif_%"] = np.inf
                else:
                    self.final_metrics["hv_dif_%"] = ((abs(self.orig_hv - self.regularity_hv)) / self.orig_hv) * 100
                self.final_metrics["igd_plus"] = self.igd_plus.do(self.F)

                self.print(f"Final IGD+ value: {'{:.2e}'.format(self.final_metrics['igd_plus'])}")
                self.print(f"Hyper-volume difference: "f"{'{:.2e}'.format(self.final_metrics['hv_dif_%'])}")

                # only keep the non-dominated solutions
                survived_pop = list(NonDominatedSorting().do(self.F)[0])
                self.X = self.X[survived_pop, :]
                self.F = self.F[survived_pop, :]

                # display two random population members
                self.print("\n Example of two population members:")
                np.random.seed(self.seed)
                ids = np.random.choice(self.X.shape[0], 2)
                table_header = ["Id"] + [str(i + 1) for i in range(self.problem_args["dim"])]
                table_data = np.zeros((2, self.problem_args["dim"] + 1))

                table_data[0, 0], table_data[0, 1:] = ids[0], self.X[ids[0], :]
                table_data[1, 0], table_data[1, 1:] = ids[1], self.X[ids[1], :]
                self.print()
                self.print(tabulate(table_data, headers=table_header))
                self.print()

                # check the applicability of the final regularity principle
                self.proxy_regular_X, self.proxy_regular_F = self._check_final_regularity(n_points=10000)

                regularity_principle = {
                    "non_fixed_independent_vars": self.regularity.non_fixed_independent_vars,
                    "non_fixed_dependent_vars": self.regularity.non_fixed_dependent_vars,
                    "non_fixed_orphan_vars": self.regularity.non_fixed_orphan_vars,
                    "coef_list": self.regularity.non_fixed_final_reg_coef_list,
                    "lb": self.regularity.lb,
                    "ub": self.regularity.ub,
                    "fixed_vars": self.regularity.fixed_vars,
                    "fixed_vals": self.regularity.fixed_vals,
                    "problem_config": self.problem_args
                }

                # save regularity principle
                pickle.dump(regularity_principle,
                            open(f"{config.BASE_PATH}/{self.result_storage}/regularity_principle.pickle",
                                 "wb"))

            else:
                self.proxy_regular_F = None
                self.proxy_regular_X = None
                self.regularity_hv = 0
                self.final_metrics["hv_dif_%"] = np.inf
                self.final_metrics["igd_plus"] = 0

        else:
            self.proxy_regular_F = None
            self.proxy_regular_X = None
            self.regularity_hv = 0
            self.final_metrics["hv_dif_%"] = np.inf
            self.final_metrics["igd_plus"] = 0

        self.print("done")

    def run_NSGA(self, problem, NSGA_settings):
        # run the NSGA2 over the problem
        if self.problem_args["n_obj"] == 2:
            # if we are dealing with a 2-objective problem, use NSGA2
            self.print("Running NSGA2..")
            algorithm = NSGA2(pop_size=NSGA_settings["pop_size"],
                              n_offsprings=NSGA_settings["n_offsprings"],
                              sampling=get_sampling("real_random"),
                              crossover=get_crossover("real_sbx", prob=NSGA_settings["sbx_prob"],
                                                      eta=NSGA_settings["sbx_eta"]),
                              mutation=get_mutation("real_pm", eta=NSGA_settings["mut_eta"]),
                              seed=self.seed,
                              eliminate_duplicate=True,
                              verbose=self.verbose)

        elif self.problem_args["n_obj"] > 2:
            # for working with many-objective problems, use NSGA3
            algorithm = NSGA3(pop_size=NSGA_settings["pop_size"],
                              n_offsprings=NSGA_settings["n_offsprings"],
                              sampling=get_sampling("real_random"),
                              crossover=get_crossover("real_sbx", prob=NSGA_settings["sbx_prob"],
                                                      eta=NSGA_settings["sbx_eta"]),
                              mutation=get_mutation("real_pm", eta=NSGA_settings["mut_eta"]),
                              ref_dirs=NSGA_settings["ref_dirs"],
                              seed=self.seed,
                              eliminate_duplicate=True,
                              ideal_point=NSGA_settings["ideal_point"],
                              nadir_point=NSGA_settings["nadir_point"],
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
                       verbose=False,
                       save_history=True)

        return res

    def find_regularity_clusters(self):
        # define the way to find the regularity features
        non_fixed_vars = []
        fixed_clusters = []

        # stage 1 - remove non-fixed variables
        num_features = self.X.shape[1]
        pop_spread = (np.max(self.X, axis=0) - np.min(self.X, axis=0)) / (np.array(self.ub) - np.array(self.lb))
        remaining_cluster_list = list(np.arange(num_features))

        # plot the deviations of variables across population
        fig = plt.figure(figsize=(8, 5))
        dim = self.X.shape[1]

        for i in range(self.X.shape[0]):
            plt.scatter(np.arange(dim), self.X[i, :])

        plt.xticks(np.arange(dim), labels=[f"$x_{i + 1} ({str(round(spread, 3))})$" for i, spread in enumerate(
            pop_spread)])
        plt.xlabel("Variables (Corresponding Spread)", fontsize=15)

        plt.tick_params(axis="x", labelsize=10, labelrotation=40)
        plt.tick_params(axis="y", labelsize=10, labelrotation=20)

        if self.verbose:
            plt.show()

        if self.save_img:
            fig.savefig(
                f"{config.BASE_PATH}/{self.result_storage}/variable_spread_pf.pdf", format="pdf")

        # find out the non-fixed variables
        for i in range(self.X.shape[1]):
            if self._is_random(self.X[:, i], i, self.lb[i], self.ub[i], self.delta)[0]:
                non_fixed_vars.append(i)
                remaining_cluster_list.remove(i)

        fixed_cluster = remaining_cluster_list
        return non_fixed_vars, fixed_cluster

    def fixed_regularity(self):
        # find the regular population
        self.X = self._non_regularity_repair(self.X)

        if len(self.X.shape) == 1:
            self.X = self.X.reshape(1, self.X.shape[0])

        self.F = self.evaluate(self.X, self.problem_args)

    def _identify_unused_vars(self,
                              coef_list):
        # identify unused dependent and independent variables
        coef_arr = np.array(coef_list)
        coef_mask = abs(coef_arr) > 0
        unused_dep = np.sum(coef_mask, axis=1) == 0
        unused_indep_degree = np.sum(coef_mask, axis=0) == 0
        unused_cases_indep = [0] * len(self.non_fixed_independent_vars)

        for i, degree_list in enumerate(self.non_fixed_degree_list):
            if unused_indep_degree[i]:
                for j, degree in enumerate(degree_list):
                    if degree >= 1:
                        unused_cases_indep[j] += 1

        max_involvement = len(self.non_fixed_degree_list) - len(np.where(np.array(self.non_fixed_degree_list)[:,
                                                                         0] == 0)[0])
        unused_dep_idx = list(np.where(unused_dep)[0])
        unused_indep_idx = list(np.where(np.array(unused_cases_indep) == max_involvement)[0])
        return unused_dep_idx, unused_indep_idx

    def non_fixed_regularity(self):
        if len(self.non_fixed_vars) > 1:
            # enforce regularity in non-fixed variables
            # sort the non-fixed variables according to decreasing order of correlation sum
            corr_var = pd.DataFrame(self.X[:, self.non_fixed_vars]).corr()
            sum_corr_var = np.array(np.sum(abs(corr_var), axis=1) - 1)
            self.non_fixed_vars = list(np.array(self.non_fixed_vars)[np.argsort(-sum_corr_var)])
            num_dependent_vars = int(np.floor(self.non_fixed_dependency_percent * len(self.non_fixed_vars)))
            num_independent_vars = len(self.non_fixed_vars) - num_dependent_vars

            # if number of independent variables more than M-1, adjust
            if num_independent_vars >= self.problem_args["n_obj"]:
                num_dependent_vars += num_independent_vars - self.problem_args["n_obj"] + 1
                num_independent_vars = self.problem_args["n_obj"] - 1

            if num_dependent_vars == 0:
                self.non_fixed_dependent_vars = []
                self.non_fixed_independent_vars = []
                self.non_fixed_orphan_vars = copy.deepcopy(self.non_fixed_vars)

            # figure out the dependent and independent variables from the set of non-fixed variables
            elif len(self.non_fixed_vars) == 2:
                # if there are two variables, both of them will have the same PCC sum
                # so check the regularities and deviations from original front and then
                # decide on the independent and dependent variable clusters
                self.print("As there are two non-fixed variables, both of them are eligible to become dependent or "
                           "independent "
                           "variable.\nSo, we are checking the pareto front deviation for both the configurations and then "
                           "we'll decide which configuration to select")

                self.print("Config 1")
                reg_X_1, _ = self._non_fixed_regularity_regression([self.non_fixed_vars[0]], [self.non_fixed_vars[1]])
                self.print("Config 2")
                reg_X_2, _ = self._non_fixed_regularity_regression([self.non_fixed_vars[1]], [self.non_fixed_vars[0]])

                X_1 = np.clip(reg_X_1, self.lb, self.ub)
                X_2 = np.clip(reg_X_2, self.lb, self.ub)

                F_1 = self.evaluate(X_1, self.problem_args)
                F_2 = self.evaluate(X_2, self.problem_args)

                hv_1 = self.hv.do(self.orig_F) - self.hv.do(F_1)
                hv_2 = self.hv.do(self.orig_F) - self.hv.do(F_2)

                # the one leading to lower hyper-volume deviation should be the better alternative
                if hv_1 < hv_2:
                    self.print("Config 1 is better than Config 2")
                    self.non_fixed_independent_vars = [self.non_fixed_vars[1]]
                    self.non_fixed_dependent_vars = [self.non_fixed_vars[0]]
                else:
                    self.print("Config 2 is better than Config 1")
                    self.non_fixed_independent_vars = [self.non_fixed_vars[0]]
                    self.non_fixed_dependent_vars = [self.non_fixed_vars[1]]

            else:
                # when there are more variables in the cluster than dependency requirement
                # select the first few variables as dependent and rest as dependent
                self.non_fixed_dependent_vars = self.non_fixed_vars[0:num_dependent_vars]
                self.non_fixed_independent_vars = self.non_fixed_vars[num_dependent_vars:]

            if len(self.non_fixed_dependent_vars) > 0:
                # get the regressed regularity (for non-fixed variables we are using degree of 1)
                reg_X, regularity_reg_coef_data = self._non_fixed_regularity_regression(self.non_fixed_dependent_vars,
                                                                                        self.non_fixed_independent_vars)
                reg_X = np.clip(reg_X, self.lb, self.ub)
                reg_F = self.evaluate(reg_X, self.problem_args)
                self.non_fixed_final_reg_coef_list = np.array(regularity_reg_coef_data)[:, 1:-2]

                # Do a final check to see if there is any non-fixed variable which is unused
                # in the equations. Those variables are called orphan variables
                if self.non_fixed_dependent_vars:
                    temp_coef_list = np.array(self.non_fixed_final_reg_coef_list[:, 0:-1])
                    unused_dep_idx, unused_indep_idx = self._identify_unused_vars(temp_coef_list)
                    unused_dep_idx = sorted(unused_dep_idx, reverse=True)
                    unused_indep_idx = sorted(unused_indep_idx, reverse=True)
                    self.non_fixed_final_reg_coef_list = np.delete(self.non_fixed_final_reg_coef_list,
                                                                   unused_dep_idx, axis=0)

                    # unused dependent variables become fixed variables
                    for dep_idx in unused_dep_idx:
                        unused_dep_var = self.non_fixed_dependent_vars.pop(dep_idx)
                        self.fixed_vars.append(unused_dep_var)
                        self.non_fixed_vars.remove(unused_dep_var)

                    # unused independent variables become non-fixed orphan variables
                    rem_degree_list = []
                    for k, degree_list in enumerate(self.non_fixed_degree_list):
                        for indep_idx in unused_indep_idx:
                            if degree_list[indep_idx] > 0:
                                rem_degree_list.append(k)
                                break

                    self.non_fixed_final_reg_coef_list = np.delete(self.non_fixed_final_reg_coef_list,
                                                                   rem_degree_list, axis=1)

                    self.non_fixed_degree_list = list(np.delete(np.array(self.non_fixed_degree_list),
                                                                unused_indep_idx,
                                                                axis=1))

                    self.non_fixed_degree_list = list(np.delete(np.array(self.non_fixed_degree_list),
                                                                rem_degree_list,
                                                                axis=0))

                    for degree_list in self.non_fixed_degree_list:
                        degree_list = list(degree_list)

                    for indep_idx in unused_indep_idx:
                        indep_var = self.non_fixed_independent_vars.pop(indep_idx)
                        self.non_fixed_orphan_vars.append(indep_var)
        else:
            self.non_fixed_orphan_vars = self.non_fixed_vars

        # display the dependent a   nd independent variables
        self.print(f"Dependent Variables: {self.non_fixed_dependent_vars}")
        self.print(f"Independent Variables: {self.non_fixed_independent_vars}")
        self.print(f"Orphan Variables: {self.non_fixed_orphan_vars}")

        # self.non_fixed_orphan_vars = self.non_fixed_dependent_vars + self.non_fixed_independent_vars
        # self.non_fixed_dependent_vars = []
        # self.non_fixed_independent_vars = []

    def _create_list_degrees(self,
                             max_degree=3,
                             num_vars=3,
                             cur_list=None,
                             full_list=None):
        if full_list is None:
            full_list = []
        if cur_list is None:
            cur_list = []

        if len(cur_list) == num_vars:
            # base check if the cur list involves all the variables
            full_list.append(cur_list)
            return

        cur_sum = 0 if len(cur_list) == 0 else np.sum(cur_list)
        for i in range(max_degree - cur_sum + 1):
            cur_list_copy = copy.deepcopy(cur_list)
            cur_list_copy.append(i)
            self._create_list_degrees(max_degree=max_degree,
                                      num_vars=num_vars,
                                      cur_list=cur_list_copy,
                                      full_list=full_list)

        # return the full list
        return full_list

    def _non_fixed_regularity_regression(self, non_fixed_dep_vars, non_fixed_indep_vars):
        # get the degree list for non-linear regression
        self.non_fixed_degree_list = self._create_list_degrees(max_degree=self.non_fixed_regularity_degree,
                                                               num_vars=len(non_fixed_indep_vars))
        # remove the all 0 degree list
        if self.non_fixed_degree_list is not None:
            self.non_fixed_degree_list.remove(self.non_fixed_degree_list[0])

        # function to regress in the non-fixed variables
        x = self.X[:, non_fixed_indep_vars]
        reg_X = copy.deepcopy(self.X)

        # storing data for tabular visualization
        def create_str_degree(non_fixed_indep_vars,
                              list_degree):
            list_strings = []
            for degrees in list_degree:
                cur_str = ""
                for degree_idx in range(len(degrees)):
                    cur_str += "x_" + str(non_fixed_indep_vars[degree_idx]) + "^" + str(degrees[
                                                                                            degree_idx])
                list_strings.append(cur_str)

            return list_strings

        list_string_headers = create_str_degree(non_fixed_indep_vars, self.non_fixed_degree_list)
        orig_reg_coef_data = np.zeros((len(non_fixed_dep_vars), 4 + len(self.non_fixed_degree_list)))
        regularity_reg_coef_data = np.zeros((len(non_fixed_dep_vars), 4 + len(self.non_fixed_degree_list)))
        orig_headers = ["Index"] + list_string_headers + ["Intercept"] + ["HV dif"] + ["MSE"]
        regular_headers = ["Index"] + list_string_headers + ["Intercept"] + ["HV dif"] + [
            "MSE"]

        if self.non_fixed_regularity_degree > 1:
            x_updated = None
            for degree_list in self.non_fixed_degree_list:
                cur_vals = np.sum(np.power(x, np.array(degree_list)), axis=1)
                if x_updated is None:
                    x_updated = cur_vals
                else:
                    x_updated = np.column_stack((x_updated, cur_vals))

            x = x_updated

        for id, i in enumerate(non_fixed_dep_vars):
            # for every dependent variable
            # we are finding the coefficients wrt the independent variables
            y = self.X[:, i].reshape(-1, 1)

            reg = linreg().fit(x, y)
            coef_ = reg.coef_[0]
            reg_X[:, i] = reg.predict(x)[:, 0]
            temp_X = copy.deepcopy(self.X)
            temp_X[:, i] = reg.predict(x)[:, 0]

            # for table formation
            orig_reg_coef_data[id, 0] = i
            orig_reg_coef_data[id, -3] = reg.intercept_
            new_hv = self.hv.do(
                self._normalize(self.evaluate(temp_X, self.problem_args), self.norm_F_lb, self.norm_F_ub))
            orig_reg_coef_data[id, -2] = round((abs(self.orig_hv - new_hv) / self.orig_hv) * 100, self.precision)
            orig_reg_coef_data[id, -1] = round(MSE(self.X, temp_X), self.precision)
            orig_reg_coef_data[id, 1:-3] = coef_.copy()

            # for every coefficient, try to find the closest value which
            # is a multiple of coefficient factor provided by the user
            for j in range(len(coef_)):
                # for every coefficient, fix it as a multiple of the coef_factor
                mult_factor_1 = int(coef_[j] / self.non_fixed_regularity_coef_factor)
                mult_factor_2 = mult_factor_1 + 1 if (coef_[j] > 0) else mult_factor_1 - 1
                mult_factor = mult_factor_1 if abs(
                    mult_factor_1 * self.non_fixed_regularity_coef_factor - coef_[j]) < abs(
                    mult_factor_2 *
                    self.non_fixed_regularity_coef_factor - coef_[j]) \
                    else mult_factor_2

                # round it of to the user provided precision
                reg.coef_[0, j] = round(self.non_fixed_regularity_coef_factor * mult_factor, self.precision)

            # round the intercept too
            mult_factor_1 = int(reg.intercept_ / self.non_fixed_regularity_coef_factor)
            mult_factor_2 = mult_factor_1 + 1 if (reg.intercept_ > 0) else mult_factor_1 - 1
            mult_factor = mult_factor_1 if abs(
                mult_factor_1 * self.non_fixed_regularity_coef_factor - reg.intercept_) < abs(
                mult_factor_2 *
                self.non_fixed_regularity_coef_factor - reg.intercept_) \
                else mult_factor_2

            reg.intercept_ = round(self.non_fixed_regularity_coef_factor * mult_factor, self.precision)

            # final regressed version
            reg_X[:, i] = reg.predict(x)[:, 0]

            temp_X = copy.deepcopy(self.X)
            temp_X[:, i] = reg.predict(x)[:, 0]

            # for table formation
            regularity_reg_coef_data[id, 0] = i
            regularity_reg_coef_data[id, -3] = reg.intercept_
            new_hv = self.hv.do(
                self._normalize(self.evaluate(temp_X, self.problem_args), self.norm_F_lb, self.norm_F_ub))
            regularity_reg_coef_data[id, -2] = round((abs(self.orig_hv - new_hv) / self.orig_hv) * 100, self.precision)
            regularity_reg_coef_data[id, -1] = round(MSE(self.X, temp_X), self.precision)
            regularity_reg_coef_data[id, 1:-3] = reg.coef_.copy()

            self.print()
            self.print(tabulate(orig_reg_coef_data, headers=orig_headers))
            self.print()
            self.print(tabulate(regularity_reg_coef_data, headers=regular_headers))

        return reg_X, regularity_reg_coef_data

    def _normalize(self, x, lb, ub):
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

    def _denormalize(self, x, lb, ub):
        # function to denormalize a value between 0 and 1
        # to a value between lb and ub
        new_x = copy.deepcopy(x)

        if len(new_x.shape) == 1:
            new_x = np.array([new_x])

        for i in range(new_x.shape[1]):
            new_x[:, i] = lb[i] + (new_x[:, i] * (ub[i] - lb[i]))

        if new_x.shape[0] == 1:
            # converting a single array back to a 1D array
            new_x = new_x[0, :]
        return new_x

    def _is_random(self, x, i, lb, ub, delta):
        # function to predict if a variable is random
        min_var, max_var = np.min(x), np.max(x)
        spread = (max_var - min_var) / (ub - lb)
        if spread <= delta:
            return False, None

        n_bins = self.n_rand_bins
        bin_counts = binned_statistic(x, x, bins=n_bins, range=(lb, ub), statistic="count")[0]
        bins_filled = np.sum(bin_counts >= 1)
        filled_fraction = bins_filled / n_bins

        fig = plt.figure()
        n, bins, patches = plt.hist(x, range=(lb, ub), bins=n_bins, edgecolor='black', linewidth=1.2)
        ticks = [patch.get_x() + patch.get_width() / 2 for patch in patches]
        plt.xticks(ticks, range(n_bins))
        # plt.title(f"Variable: $X_{i+1}$, n_bins: {n_bins}, filled_bins: {filled_fraction*100}%")
        if self.save_img:
            plt.savefig(
                f"{config.BASE_PATH}/{self.result_storage}/variable_{i + 1}_histogram.pdf", format="pdf")
        if self.verbose:
            plt.show()

        if filled_fraction >= 0.5:
            return True, filled_fraction
        else:
            return False, filled_fraction

    def _non_regularity_repair(self, X_apply):
        X_copy = copy.deepcopy(X_apply)
        self.fixed_vals = np.round(np.mean(X_apply[:, self.fixed_vars], axis=0), self.precision)
        X_copy[:, self.fixed_vars] = self.fixed_vals

        return X_copy

    def _find_precision(self, num):
        # function to find precision of a floating point number
        num_str = str(num)
        return max(len(num_str) - num_str.find(".") - 1, 0)

    def re_optimize(self):
        # use the same problem setting to run NSGA2 another time to handle cv
        new_problem_args = copy.deepcopy(self.problem_args)
        new_NSGA_settings = copy.deepcopy(self.NSGA_settings)

        # formulate the new shorter problem
        # dim of the new problem is the number of non-fixed independent and non-fixed orphan variables
        new_problem_args["dim"] = len(self.non_fixed_independent_vars) + len(self.non_fixed_orphan_vars)
        combined_non_fixed_vars = self.non_fixed_independent_vars + self.non_fixed_orphan_vars

        # take the corresponding lb and ubs
        new_problem_args["lb"] = self.pareto_lb if len(self.pareto_lb) == 1 else [
            self.pareto_lb[i] for i in combined_non_fixed_vars]
        new_problem_args["ub"] = self.pareto_ub if len(self.pareto_ub) == 1 else [
            self.pareto_ub[i] for i in combined_non_fixed_vars]

        # save the mapping of the variables of the smaller problem to the large problem
        new_problem_args["non_fixed_variable_mapper"] = {
            "non_fixed_independent_vars": list(np.arange(len(self.non_fixed_independent_vars))),
            "non_fixed_orphan_vars": list(
                np.arange(len(self.non_fixed_independent_vars), len(combined_non_fixed_vars))),
        }

        # set the regularity information so that every time it enforces the regularity on its population
        new_problem_args["fixed_vars"] = self.fixed_vars
        new_problem_args["fixed_vals"] = self.fixed_vals
        new_problem_args["non_fixed_vars"] = self.non_fixed_vars
        new_problem_args["non_fixed_dependent_vars"] = self.non_fixed_dependent_vars
        new_problem_args["non_fixed_dependent_lb"] = self.pareto_lb if len(self.pareto_lb) == 1 else [
            self.pareto_lb[i] for i in self.non_fixed_dependent_vars]
        new_problem_args["non_fixed_dependent_ub"] = self.pareto_ub if len(self.pareto_ub) == 1 else [
            self.pareto_ub[i] for i in self.non_fixed_dependent_vars]
        new_problem_args["non_fixed_independent_vars"] = self.non_fixed_independent_vars
        new_problem_args["non_fixed_orphan_vars"] = self.non_fixed_orphan_vars
        new_problem_args["non_fixed_orphan_lb"] = self.pareto_lb if len(self.pareto_lb) == 1 else [
            self.pareto_lb[i] for i in self.non_fixed_orphan_vars]
        new_problem_args["non_fixed_orphan_ub"] = self.pareto_ub if len(self.pareto_ub) == 1 else [
            self.pareto_ub[i] for i in self.non_fixed_orphan_vars]

        # regularity_enforcement is True when we are dealing with the smaller problem
        new_problem_args["regularity_enforcement"] = True
        new_problem_args["regularity_enforcement_process"] = self._final_regularity_enforcement()

        # add constraints on the bounds of the dependent variables
        # following the regularitys may take some of them out of bounds
        new_problem_args["n_constr"] = self.problem_args["n_constr"] + (len(self.non_fixed_dependent_vars) * 2)

        # get the smaller problem
        new_problem = get_problem(self.problem_name, new_problem_args)

        # rerun NSGA2
        new_res = self.run_NSGA(new_problem, new_NSGA_settings)

        if new_res.X is not None:
            I = NonDominatedSorting().do(new_res.F)
            new_res.X = new_res.X[I[0], :]
            new_res.F = new_res.F[I[0], :]

            # get the solutions from the smaller problem and map them to the original problem
            new_X = np.zeros((new_res.X.shape[0], self.problem_args["dim"]))
            new_X[:, self.non_fixed_independent_vars] = new_res.X[:, new_problem_args["non_fixed_variable_mapper"][
                                                                         "non_fixed_independent_vars"]]
            new_X[:, self.non_fixed_orphan_vars] = new_res.X[:, new_problem_args["non_fixed_variable_mapper"][
                                                                    "non_fixed_orphan_vars"]]
            new_X = self._final_regularity_enforcement()(new_X)

            # set the X and evaluate the regular population
            self.X = new_X
            self.F = self.evaluate(self.X, self.problem_args)

        else:
            self.X = None
            self.F = None
            self.print("No feasible solution found with this regularity...")

    def _final_regularity_enforcement(self):
        # returns the function that enforces the final regularity to a given X
        def _regularity_enforcement(X):
            # enforce the final regularity to any vector of same size
            new_X = copy.deepcopy(X)

            # for fixed variables fix them to the found values
            new_X[:, self.fixed_vars] = self.fixed_vals

            # for non-fixed variables use the equation found in the process
            if self.non_fixed_dependent_vars and self.non_fixed_independent_vars:
                for i, dep_idx in enumerate(self.non_fixed_dependent_vars):
                    new_X[:, dep_idx] = 0
                    for j, list_degree in enumerate(self.non_fixed_degree_list):
                        new_X[:, dep_idx] += self.non_fixed_final_reg_coef_list[i][j] * \
                                             np.sum(np.power(new_X[:, self.non_fixed_independent_vars],
                                                             np.array(list_degree)), axis=1)

                    new_X[:, dep_idx] += self.non_fixed_final_reg_coef_list[i][-1]

            return new_X

        return _regularity_enforcement

    def _check_final_regularity(self,
                                n_points=1000):
        # create a population of random solutions and apply the regularity over them
        random_X = np.random.uniform(self.pareto_lb, self.pareto_ub, (n_points, self.problem_args["dim"]))
        regular_X = self.regularity.apply(random_X)
        regular_F, regular_G = self.evaluate(regular_X, self.problem_args, constr=True)

        if regular_G is not None:
            if len(regular_G.shape) == 1:
                regular_G = regular_G.reshape(-1, 1)
            constrained_idx = list(np.where(
                (np.sum(regular_G > 0, axis=1) == 0) * np.prod(
                    (self.pareto_lb <= regular_X) * (regular_X <= self.pareto_ub),
                    axis=1))[0])
            regular_X = regular_X[constrained_idx, :]
            regular_F = regular_F[constrained_idx, :]

        if len(regular_X) == 0:
            return None, None

        required_pop_size = min(self.X.shape[0], regular_X.shape[0])
        regular_pop = pop_from_array_or_individual(regular_X)
        regular_pop.set("F", regular_F)

        if self.problem_args["n_obj"] == 2:
            # for 2-objective problems use rank and crowding survival
            regular_pop = RankAndCrowdingSurvival()._do(self.problem,
                                                        regular_pop,
                                                        n_survive=required_pop_size)
            regular_X = regular_pop.get("X")
            regular_F = regular_pop.get("F")

        elif self.problem_args["n_obj"] > 2:
            # for >2 objectives use reference direction based survival
            regular_pop = ReferenceDirectionSurvival(self.NSGA_settings["ref_dirs"])._do(self.problem,
                                                                                         regular_pop,
                                                                                         n_survive=required_pop_size)
            regular_X = regular_pop.get("X")
            regular_F = regular_pop.get("F")

        else:
            print("[Error!] Wrong dimensions")

        return regular_X, regular_F
