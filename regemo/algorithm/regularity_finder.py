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
from pymoo.visualization.pcp import PCP
from pymoo.visualization.scatter import Scatter

from sklearn.linear_model import LinearRegression as linreg
from sklearn.metrics import mean_squared_error as MSE

import copy
import sys
from tabulate import tabulate

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import binned_statistic
import pandas as pd

plt.rcParams.update({'font.size': 15})

class Regularity_Finder():
    def __init__(self,
                 X,
                 F,
                 problem_args,
                 non_rand_regularity_degree=1,
                 rand_regularity_coef_factor=0.1,
                 rand_regularity_dependency=0,
                 rand_regularity_MSE_threshold=1,
                 non_rand_regularity_MSE_threshold=1,
                 precision=2,
                 NSGA_settings=None,
                 clustering_config=None,
                 save_img=True,
                 result_storage="/.",
                 verbose=True,
                 pf_cluster_num=0,
                 num_clusters=1,
                 seed=0):
        """
        :param X: population of solutions
        :param F: evluation scores for the population
        :param problem_args: arguments for the problem
        :param non_rand_regularity_degree: degree for non-random regularity fitting
        :param rand_regularity_coef_factor: coefficient of multiplication for rand variable regularity
        :param rand_regularity_dependency: max number of dependent random variables
        :param rand_regularity_MSE_threshold: the threshold for keeping rand pattern
        :param non_rand_regularity_MSE_threshold: the threshold for keeping non-rand pattern
        :param precision: precision for the floating point values (used for simplification of the regularity)
        :param NSGA_settings: algorithm settings for NSGA2/NSGA3
        :param clustering_config: configuration for clustering in the population
        :param save_img: if we want to save the images
        :param result_storage: the storage directory for the results
        :param verbose: whether to print the intermediate statements
        :param pf_cluster_num: max number of clusters in the pareto front
        :param num_clusters: total number of clusters present in the entire population (used for binning)
        :param seed: random seed
        """

        super().__init__()

        # set problem-specific information
        self.problem_name = problem_args["name"]
        self.problem_args = problem_args
        self.seed = seed
        self.non_rand_regularity_degree = non_rand_regularity_degree
        self.rand_regularity_coef_factor = rand_regularity_coef_factor
        self.rand_regularity_dependency = rand_regularity_dependency
        self.rand_regularity_MSE_threshold = rand_regularity_MSE_threshold
        self.non_rand_regularity_MSE_threshold = non_rand_regularity_MSE_threshold    # non-rand threshold
        self.NSGA_settings = NSGA_settings
        self.clustering_config = clustering_config

        # get the problem utilities
        self.problem = get_problem(self.problem_name, problem_args)
        self.evaluate = get_problem(self.problem_name, problem_args, class_required=False)

        # initialize the regularity algorithm parameters
        self.pf_cluster_num = pf_cluster_num
        self.rand_cluster = None
        self.non_rand_cluster = None
        self.non_rand_vars = None
        self.non_rand_vals = None
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
        self.rand_final_reg_coef_list = []
        self.rand_dependent_vars = []
        self.rand_independent_vars = []
        self.rand_complete_vars = []
        self.result_storage = result_storage
        self.igd_plus = None
        self.hv = None
        self.precision = precision
        self.save_img = save_img
        self.verbose = verbose
        self.print = verboseprint(self.verbose)
        self.regularity = None
        self.final_metrics = {}
        self.num_clusters = num_clusters

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

        default_clustering_config = {
            "min_cluster_size": 3,
            "max_clusters": 4,
            "MSE_threshold": 0.0002
        }

        if self.NSGA_settings is None:
            # if NSGA_settings is not specified, use the default one
            self.NSGA_settings = default_NSGA_settings
        else:
            # if NSGA_settings is specified, but all the parameters are not properly set
            # set the remaining parameters from the default setting
            missing_keys = list(np.setdiff1d(list(default_NSGA_settings.keys()), list(self.NSGA_settings.keys())))
            for i in missing_keys:
                self.NSGA_settings[i] = default_NSGA_settings[i]

        if self.clustering_config is None:
            # if clustering_config is not specified, use the default one
            self.clustering_config = default_clustering_config
        else:
            # if clustering_config is specified, but all the parameters are not properly set
            # set the remaining parameters from the default setting
            missing_keys = list(np.setdiff1d(list(default_clustering_config.keys()), list(self.clustering_config.keys())))
            for i in missing_keys:
                self.clustering_config[i] = default_clustering_config[i]

    def run(self):
        # main code to run the regularity enforcing algorithm
        self.orig_X, self.orig_F = copy.deepcopy(self.X), copy.deepcopy(self.F)

        # set the performance indicators
        self.igd_plus = get_performance_indicator("igd+", self.orig_F)
        self.hv = get_performance_indicator("hv", ref_point=np.ones(self.problem_args["n_obj"]))

        # set the initial metrics
        self.norm_F_lb = np.min(self.orig_F, axis=0)
        self.norm_F_ub = np.max(self.orig_F, axis=0)

        self.norm_orig_F = self._normalize(self.orig_F, self.norm_F_lb, self.norm_F_ub)
        self.orig_hv = self.hv._do(self.norm_orig_F)

        # clustering
        self.rand_cluster, self.non_rand_cluster = self.find_regularity_clusters()
        self.non_rand_vars = sum(self.non_rand_cluster, [])

        # regularity enforcement in non-random variables
        if self.non_rand_vars:
            self.print("================================================")
            self.print("Searching for regularity inside non-rand variables...")
            self.print("================================================")
            self.non_rand_regularity()
            self.print("The regularity is that all the population members are having the same values for the non-rand clusters "
                  "variables\n where each cluster follows a non-decreasing curve with degree " +
                  str(self.non_rand_regularity_degree))
            for cluster in self.non_rand_cluster:
                self.print("Cluster: ", cluster, ", Values: ", self.X[0, cluster])
            # after the non-rand values are found, save those values in non_rand_vals
            self.non_rand_vals = list(self.X[0, self.non_rand_vars])

        else:
            self.print("[INFO] The process is stopping as there's no non-random variables")

        # regularity enforcement in random variables
        if len(self.rand_cluster) > 0:
            self.print("\n================================================")
            self.print("Searching for regularity inside random variables...")
            self.print("================================================")
            self.rand_regularity()
            if self.rand_dependent_vars:
                self.print("The regularity is that the slopes of the projection of the variables with respect to the index variable "
                      "are always multiples of " + str(self.rand_regularity_coef_factor))


        # save the regularity to a regularity object
        self.regularity = Regularity(dim=self.problem_args["dim"],
                               lb=self.lb,
                               ub=self.ub,
                               non_rand_cluster=self.non_rand_cluster,
                               non_rand_vals=self.non_rand_vals,
                               rand_cluster=self.rand_cluster,
                               rand_dependent_vars=self.rand_dependent_vars,
                               rand_independent_vars=self.rand_independent_vars,
                               rand_complete_vars=self.rand_complete_vars,
                               rand_final_reg_coef_list=self.rand_final_reg_coef_list,
                               degree=self.non_rand_regularity_degree,
                               precision=self.precision)

        if self.verbose:
            self.regularity.display(self.orig_X, self.lb, self.ub)

        self.X = self.regularity.apply(self.X, self.lb, self.ub)
        self.F = self.evaluate(self.X, self.problem_args)

        # plot the regular front
        plot = Scatter(labels="F", legend=True, angle=self.problem_args["visualization_angle"])
        plot = plot.add(self.orig_F, color="blue", marker="o", s=15, label="Original Efficient Front")
        plot = plot.add(self.F, color="red", marker="*", s=40, label="Regular Efficient Front")

        # plot.title = "Regular Efficient Front (Before Re-optimization)"

        if self.verbose:
            plot.show()

        if self.save_img:
            plot.save(f"{config.BASE_PATH}/{self.result_storage}/regular_efficient_front_pre_reopt_cluster_{self.pf_cluster_num+ 1}.jpg")

        # Re-optimization
        if len(self.rand_cluster):
            self.print("\nRe-optimizing the population after regularity enforcement...")
            self.re_optimize()

        # final metric calculation
        self.norm_F = self._normalize(self.F, self.norm_F_lb, self.norm_F_ub)
        if self.norm_F.ndim == 1:
            self.norm_F = self.norm_F.reshape(1, -1)

        # save the final metrics
        self.regularity_hv = self.hv._do(self.norm_F)

        if self.regularity_hv == 0:
            # when it converges to 1 point
            self.final_metrics["hv_dif_%"] = np.inf
        else:
            self.final_metrics["hv_dif_%"] = ((abs(self.orig_hv - self.regularity_hv)) / self.orig_hv) * 100
        self.final_metrics["igd_plus"] = self.igd_plus._do(self.F)
        # self.final_metrics["MSE"] = MSE(np.mean(self.orig_X, axis=0), np.mean(self.X, axis=0))

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
        table_header = ["Id"] + [str(i) for i in range(self.problem_args["dim"])]
        table_data = np.zeros((2, self.problem_args["dim"]+1))

        table_data[0, 0], table_data[0, 1:] = ids[0], self.X[ids[0], :]
        table_data[1, 0], table_data[1, 1:] = ids[1], self.X[ids[1], :]
        self.print()
        self.print(tabulate(table_data, headers=table_header))
        self.print()

        # get a pcp plot of the final population
        plot = PCP(
            # title=("Final Regular Population", {'pad': 30}),
                   labels="X"
                   )
        plot.normalize_each_axis = False

        plot.set_axis_style(color="grey", alpha=0.5)
        plot.add(self.X)

        if self.save_img:
            plot.save(f"{config.BASE_PATH}/{self.result_storage}/PCP_final_population.jpg")

        if self.verbose:
            plot.show()

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

    def non_rand_regularity(self):
        # find the regular population
        self.X = np.clip(self._regularity_repair(self.X, self.non_rand_cluster, self.non_rand_regularity_degree),
                               self.lb, self.ub)

        if len(self.X.shape) == 1:
            self.X = self.X.reshape(1, self.X.shape[0])

        self.F = self.evaluate(self.X, self.problem_args)

    def rand_regularity(self):

        if len(self.rand_cluster)>1:
            # enforce regularity in random variables
            # sort the random variables according to decreasing order of correlation sum
            corr_var = pd.DataFrame(self.X[:, self.rand_cluster]).corr()
            sum_corr_var = np.array(np.sum(abs(corr_var), axis=1) - 1)
            self.rand_cluster = list(np.array(self.rand_cluster)[np.argsort(-sum_corr_var)])
            num_dependent_vars = self.rand_regularity_dependency
            num_independent_vars = len(self.rand_cluster) - num_dependent_vars

            # figure out the dependent and independent variables from the set of random variables
            if len(self.rand_cluster) == 2:
                # if there are two variables, both of them will have the same PCC sum
                # so check the regularitys and deviations from original front and then
                # decide on the independent and dependent variable clusters
                self.print("As there are two random variables, both of them are eligible to become dependent or "
                           "independent "
                      "variable.\nSo, we are checking the pareto front deviation for both the configurations and then "
                      "we'll decide which configuration to select")

                self.print("Config 1")
                reg_X_1, _ = self._rand_regularity_regression([self.rand_cluster[0]], [self.rand_cluster[1]])
                self.print("Config 2")
                reg_X_2, _ = self._rand_regularity_regression([self.rand_cluster[1]], [self.rand_cluster[0]])

                X_1 = np.clip(reg_X_1, self.lb, self.ub)
                X_2 = np.clip(reg_X_2, self.lb, self.ub)

                F_1 = self.evaluate(X_1, self.problem_args)
                F_2 = self.evaluate(X_2, self.problem_args)

                hv_1 = self.hv._do(self.orig_F) - self.hv._do(F_1)
                hv_2 = self.hv._do(self.orig_F) - self.hv._do(F_2)

                # the one leading to lower hyper-volume deviation should be the better alternative
                if hv_1 < hv_2:
                    self.print("Config 1 is better than Config 2")
                    self.rand_independent_vars = [self.rand_cluster[1]]
                    self.rand_dependent_vars = [self.rand_cluster[0]]
                else:
                    self.print("Config 2 is better than Config 1")
                    self.rand_independent_vars = [self.rand_cluster[0]]
                    self.rand_dependent_vars = [self.rand_cluster[1]]

            elif len(self.rand_cluster) > self.rand_regularity_dependency:
                # when there are more variables in the cluster than dependency requirement
                # select the the first few variables as independent and rest as dependent
                self.rand_independent_vars = self.rand_cluster[0:num_independent_vars]
                self.rand_dependent_vars = self.rand_cluster[num_independent_vars:]

            else:
                # when the dependency requirement is higher than the number of variables in the cluster
                # map the dependency to the closest applicable value
                self.print("[INFO] The specified rand regularity dependency is not possible...")
                self.rand_regularity_dependency = np.clip(self.rand_regularity_dependency, 0, len(self.rand_cluster)-1)
                self.print(f"The closest dependency possible is: {self.rand_regularity_dependency}")
                num_dependent_vars = self.rand_regularity_dependency
                num_independent_vars = len(self.rand_cluster) - num_dependent_vars

                self.rand_independent_vars = self.rand_cluster[0:num_independent_vars]
                self.rand_dependent_vars = self.rand_cluster[num_independent_vars:]

            # get the regressed regularity (for random variables we are using degree of 1)
            reg_X, regularityed_reg_coef_data = self._rand_regularity_regression(self.rand_dependent_vars, self.rand_independent_vars)
            reg_X = np.clip(reg_X, self.lb, self.ub)
            reg_F = self.evaluate(reg_X, self.problem_args)

            # if normalized MSE < threshold,
            #   accept
            rand_diff = abs(reg_X[:, self.rand_dependent_vars] - self.X[:, self.rand_dependent_vars])
            normalized_diff = (rand_diff - np.array(self.lb)[self.rand_dependent_vars])/(np.array(self.ub)[
                                                                                             self.rand_dependent_vars]-
                                                                                         np.array(self.lb)[self.rand_dependent_vars])
            normalized_MSE = np.mean(np.sqrt(np.sum(normalized_diff ** 2, axis=1) / len(self.rand_dependent_vars)))

            if normalized_MSE <= self.rand_regularity_MSE_threshold:
                # get the regressed population and evaluate them
                self.rand_final_reg_coef_list = np.array(regularityed_reg_coef_data)[:, 1:-2]

            else:
                self.print("The metrics exceeded the threshold... Not enforcing the random regularity")
                self.rand_dependent_vars = []
                self.rand_independent_vars = []
                self.rand_complete_vars = self.rand_cluster

            # Do a final check to see if there is any random variable which is unused
            # in the equations. Those variables are called complete random variables
            if self.rand_dependent_vars:
                temp_coef_list = np.array(self.rand_final_reg_coef_list[:, 0:-1])
                rand_indep_var_unutilized = np.prod(temp_coef_list == 0, axis=0)
                complete_rand_indices = list(np.where(rand_indep_var_unutilized != 0)[0])

                if complete_rand_indices:
                    indep_rand_indices = np.setdiff1d(np.arange(len(self.rand_independent_vars)), complete_rand_indices)
                    self.print(f"Independent random indices: {indep_rand_indices}, complete random indices: {complete_rand_indices}, independent variables: {self.rand_independent_vars}")
                    self.rand_complete_vars = list(np.array(self.rand_independent_vars)[complete_rand_indices])
                    self.rand_independent_vars = list(np.array(self.rand_independent_vars)[indep_rand_indices])
                    useful_params = np.setdiff1d(np.arange(self.rand_final_reg_coef_list.shape[1]), complete_rand_indices)
                    self.rand_final_reg_coef_list = self.rand_final_reg_coef_list[:, useful_params]

                # if there is no independent variable, make the dependent variables complete random
                if not self.rand_independent_vars:
                    self.rand_complete_vars = self.rand_complete_vars + self.rand_dependent_vars
                    self.rand_dependent_vars = []

        else:
            self.rand_complete_vars = self.rand_cluster
            normalized_MSE = 0.0

        # change the bounds
        final_rand_vars = self.rand_complete_vars + self.rand_independent_vars
        self.lb = np.array(self.lb)
        self.lb[final_rand_vars] = np.round(np.min(self.X[:, final_rand_vars], axis=0), self.precision)
        self.lb = list(self.lb)

        self.ub = np.array(self.ub)
        self.ub[final_rand_vars] = np.round(np.max(self.X[:, final_rand_vars], axis=0), self.precision)
        self.ub = list(self.ub)

        # fix ub if lb and ub becomes the same
        # Add ub[i] = lb[i] * 1.0001
        for i, x in enumerate(self.ub):
            if self.lb[i] != x:
                self.ub[i] = x
            else:
                self.ub[i] = x + (self.problem_args["ub"][i]-x)*0.01

        # display the dependent a   nd independent variables
        self.print(f"Dependent Variables: {self.rand_dependent_vars}")
        self.print(f"Independent Variables: {self.rand_independent_vars}")
        self.print(f"Complete Random Variables: {self.rand_complete_vars}")

        # print the MSE between reg_X and X
        self.print(f"The normalized MSE between regressed and original X is: {normalized_MSE}")

    def _rand_regularity_regression(self, rand_dep_vars, rand_indep_vars):
        # function to regress in the random variables
        x = self.X[:, rand_indep_vars]
        reg_X = copy.deepcopy(self.X)

        # storing data for tabular visualization
        orig_reg_coef_data = np.zeros((len(rand_dep_vars), 4 + len(rand_indep_vars)))
        regularityed_reg_coef_data = np.zeros((len(rand_dep_vars), 4 + len(rand_indep_vars)))
        orig_headers = ["Index"] + [str(idx) for idx in rand_indep_vars] + ["Intercept"] + ["HV dif"] + ["MSE"]
        regularityed_headers = ["Index"] + [str(idx) + "'" for idx in rand_indep_vars] + ["Intercept"] + ["HV dif"] + [
            "MSE"]

        for id, i in enumerate(rand_dep_vars):
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
            new_hv = self.hv._do(self._normalize(self.evaluate(temp_X, self.problem_args), self.norm_F_lb, self.norm_F_ub))
            orig_reg_coef_data[id, -2] = round((abs(self.orig_hv - new_hv)/self.orig_hv) * 100, self.precision)
            orig_reg_coef_data[id, -1] = round(MSE(self.X, temp_X), self.precision)
            orig_reg_coef_data[id, 1:-3] = coef_.copy()

            # for every coefficient, try to find the closest value which
            # is a multiple of coefficient factor provided by the user
            for j in range(len(coef_)):
                # for every coefficient, fix it as a multiple of the coef_factor
                mult_factor_1 = int(coef_[j] / self.rand_regularity_coef_factor)
                mult_factor_2 = mult_factor_1 + 1 if (mult_factor_1 > 0) else mult_factor_1 - 1
                mult_factor = mult_factor_1 if abs(mult_factor_1 * self.rand_regularity_coef_factor - coef_[j]) < abs(
                    mult_factor_2 *
                    self.rand_regularity_coef_factor - coef_[j]) \
                    else mult_factor_2

                # round it of to the user provided precision
                reg.coef_[0, j] = round(self.rand_regularity_coef_factor * mult_factor, self.precision)

            # round the intercept too
            reg.intercept_ = np.round(reg.intercept_, self.precision)

            # final regressed version
            reg_X[:, i] = reg.predict(x)[:, 0]

            temp_X = copy.deepcopy(self.X)
            temp_X[:, i] = reg.predict(x)[:, 0]

            # for table formation
            regularityed_reg_coef_data[id, 0] = i
            regularityed_reg_coef_data[id, -3] = reg.intercept_
            new_hv = self.hv._do(
                self._normalize(self.evaluate(temp_X, self.problem_args), self.norm_F_lb, self.norm_F_ub))
            regularityed_reg_coef_data[id, -2] = round((abs(self.orig_hv - new_hv)/self.orig_hv) * 100, self.precision)
            regularityed_reg_coef_data[id, -1] = round(MSE(self.X, temp_X), self.precision)
            regularityed_reg_coef_data[id, 1:-3] = reg.coef_.copy()

            self.print()
            self.print(tabulate(orig_reg_coef_data, headers=orig_headers))
            self.print()
            self.print(tabulate(regularityed_reg_coef_data, headers=regularityed_headers))

        return reg_X, regularityed_reg_coef_data

    def _compute_regularityed_MSE(self, X, clusters):
        # find the MSE between the mean vector and its regularityed version
        mean_X = np.mean(X, axis=0)
        reg_X = self._regularity_repair(mean_X, [clusters], self.non_rand_regularity_degree)

        return MSE(mean_X, reg_X)

    def _normalize(self, x, lb, ub):
        # function to normalize x between 0 and 1
        new_x = copy.deepcopy(x)

        if new_x.ndim == 1:
            new_x = np.array([new_x])

        for i in range(new_x.shape[1]):
            new_x[:, i] = (new_x[:, i] - lb[i])/(ub[i] - lb[i])

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

    def _cluster_break(self, X, cluster, min_cluster_size):
        # function to break the non-random variables into different clusters based on the config
        cluster_pop_mean = np.mean(X, axis=0)
        repaired_cluster_pop_mean = self._regularity_repair(cluster_pop_mean, [cluster], self.non_rand_regularity_degree)
        # check the initial MSE between the actual mean X and the regressed mean X
        cur_MSE = MSE(cluster_pop_mean, repaired_cluster_pop_mean)

        # the candidate index for breaking up the cluster is the index along which
        # the regressed mean X has the maximum deviation
        dist = abs(cluster_pop_mean[cluster] - repaired_cluster_pop_mean[cluster])
        break_idx = np.argmax(dist)

        # there are two options for inclusion of the break_idx
        # it can added to the left cluster or the right
        # In [0, 1, 2, 3] if 2 is the break_point
        # option 1: [0, 1, 2] , [3]
        # option 2: [0, 1], [2, 3]
        options = []
        for i in range(2):
            # evaluate both the options and pick the best one
            cur_option_res = {}
            cur_option_res["cluster_left"] = cluster[0:break_idx + i]
            cur_option_res["cluster_right"] = cluster[break_idx + i:]
            cur_option_res["MSE"] = np.float64("inf")

            if len(cur_option_res["cluster_left"]) < min_cluster_size or \
                    len(cur_option_res["cluster_right"]) < min_cluster_size:
                cur_option_res["MSE"] = np.float64("inf")
            else:
                # repair the left_cluster and get the MSE
                repaired_cluster_pop_mean_left = \
                    self._regularity_repair(cluster_pop_mean, [cur_option_res["cluster_left"]], self.non_rand_regularity_degree)
                cur_MSE_left = MSE(cluster_pop_mean, repaired_cluster_pop_mean_left)

                # repair the right_cluster and get the MSE
                repaired_cluster_pop_mean_right = \
                    self._regularity_repair(cluster_pop_mean, [cur_option_res["cluster_right"]],
                                         self.non_rand_regularity_degree)
                cur_MSE_right = MSE(cluster_pop_mean, repaired_cluster_pop_mean_right)

                # overall MSE is the sum of both the MSEs
                cur_option_res["MSE"] = cur_MSE_right + cur_MSE_left

            options.append(cur_option_res)

        # better break is the one with least total MSE
        better_break = 0 if options[0]["MSE"] < options[1]["MSE"] else 1
        # check the decrement in MSE achieved through this break
        options[better_break]["MSE_decrement"] = cur_MSE - options[better_break]["MSE"]

        return options[better_break]

    def _is_random(self, x, i, lb, ub):
        # function to predict if a variable is random
        min_var, max_var = np.min(x), np.max(x)
        spread = (max_var - min_var)/(ub - lb)
        if spread <= (0.05/self.num_clusters):
            return False

        n_bins = 20
        bin_counts = binned_statistic(x, x, bins=n_bins, range=(lb, ub), statistic="count")[0]
        bins_filled = np.sum(bin_counts >= 1)
        filled_fraction = bins_filled/n_bins

        fig = plt.figure()
        n, bins, patches = plt.hist(x, range=(lb, ub), bins=n_bins, edgecolor='black', linewidth=1.2)
        ticks = [patch.get_x() + patch.get_width()/2 for patch in patches]
        plt.xticks(ticks, range(n_bins))
        # plt.title(f"Variable: $X_{i+1}$, n_bins: {n_bins}, filled_bins: {filled_fraction*100}%")
        if self.save_img:
            plt.savefig(f"{config.BASE_PATH}/{self.result_storage}/cluster_{self.pf_cluster_num+1}_variable_{i+1}_histogram.jpg")
        if self.verbose:
            plt.show()

        if filled_fraction >= (0.5/self.num_clusters):
            return True
        else:
            return False

    def find_regularity_clusters(self):
        # define the way to find the regularity features
        rand_cluster = []
        non_rand_clusters = []

        # stage 1 - remove random variables
        num_features = self.X.shape[1]
        pop_spread = (np.max(self.X, axis=0) - np.min(self.X, axis=0))/(np.array(self.ub) - np.array(self.lb))
        remaining_cluster_list = list(np.arange(num_features))

        # plot the deviations of variables across population
        fig = plt.figure(figsize=(10, 8))
        dim = self.X.shape[1]

        for i in range(self.X.shape[0]):
            plt.scatter(np.arange(dim), self.X[i, :])

        plt.xticks(np.arange(dim), labels=[f"$X_{i+1} ({str(round(spread, 3))})$" for i, spread in enumerate(
            pop_spread)])
        plt.xlabel("Variables (Corresponding Spread)")
        # plt.title(f"Spread of Variables across Population for {self.problem_args['name']} Problem")

        plt.tick_params(axis="x", labelsize=10, labelrotation=40)
        plt.tick_params(axis="y", labelsize=10, labelrotation=20)

        if self.verbose:
            plt.show()

        if self.save_img:
            fig.savefig(f"{config.BASE_PATH}/{self.result_storage}/variable_spread_pf_cluster_{self.pf_cluster_num+1}.jpg")

        # find out the random variables
        for i in range(self.X.shape[1]):
            if self._is_random(self.X[:, i], i, self.lb[i], self.ub[i]):
                rand_cluster.append(i)
                remaining_cluster_list.remove(i)

        # stage 2 - break the clusters from the most unstable point
        # sort the cluster indices in increasing order of mean values
        sorted_idx = np.argsort(np.median(self.X[:, remaining_cluster_list], axis=0))
        remaining_cluster_list = [list(np.array(remaining_cluster_list)[sorted_idx])]
        best_break_cluster_idx = len(remaining_cluster_list)

        while len(remaining_cluster_list) < self.clustering_config["max_clusters"] and best_break_cluster_idx != -1:
            # the breaking continues till we reach a user-specified number of clusters or
            # it is not possible to break a cluster any more
            best_break_cluster_idx = -1
            best_MSE_decrease = -np.float64("inf")
            cluster_break_results = {}

            # at any time, we can have multiple clusters of variables
            # it is important to fix which cluster to break
            # we break the cluster which results into the max MSE decrease
            for cluster_idx, cur_cluster in enumerate(remaining_cluster_list):
                if self._compute_regularityed_MSE(self.X, cur_cluster) > self.clustering_config["MSE_threshold"]:
                    cur_break_res = self._cluster_break(self.X, cur_cluster, self.clustering_config["min_cluster_size"])
                    cluster_break_results[cluster_idx] = cur_break_res

                    if cur_break_res["MSE_decrement"] > best_MSE_decrease:
                        best_MSE_decrease = cur_break_res["MSE_decrement"]
                        best_break_cluster_idx = cluster_idx

            if best_break_cluster_idx != -1:
                # if there is a cluster break resulting into some MSE_decrement
                # remove the breaking cluster from the remaining cluster list
                # add the two clusters resulting from the breaking to the list
                remaining_cluster_list.remove(remaining_cluster_list[best_break_cluster_idx])
                remaining_cluster_list.append(cluster_break_results[best_break_cluster_idx]["cluster_left"])
                remaining_cluster_list.append(cluster_break_results[best_break_cluster_idx]["cluster_right"])

        # finally the remaining cluster list contains all the non-random clusters
        # resulting into min deviation from the actual median
        # following the user-specified regularitys
        for cur_cluster in remaining_cluster_list:
            non_rand_clusters.append(cur_cluster)

        # display the random and non-random clusters found throught the process
        self.print("Rand Cluster: ", rand_cluster, ", Non-rand Clusters: ", non_rand_clusters)
        return rand_cluster, non_rand_clusters

    def _regularity_repair(self, X_apply, regularity_clusters, degree):
        # define the regularity repair procedure for non-random clusters of variables
        X = copy.deepcopy(X_apply)
        if len(X.shape) == 1:
            # linear regression needs a 2D array
            X = np.array([X])

        # take the mean of the population across the variables
        new_X = copy.deepcopy(X)
        mean_X = np.mean(X, axis=0)

        # repair the mean values based on the clusters and clip+round them
        normalized_mean_X = self._normalize(mean_X, self.lb, self.ub)
        normalized_regularityed_mean_X = fit_curve(normalized_mean_X, regularity_clusters, degree)
        regularityed_mean_X = self._denormalize(normalized_regularityed_mean_X, self.lb, self.ub)
        regularityed_mean_X = np.round(np.clip(regularityed_mean_X, self.lb, self.ub), self.precision)

        # check if the error is acceptable
        if MSE(mean_X, regularityed_mean_X) <= self.non_rand_regularity_MSE_threshold:
            # for every cluster of non-random  variables, fix all the population members
            # to corresponding repaired mean values
            for cluster in regularity_clusters:
                new_X[:, cluster] = regularityed_mean_X[cluster]
        else:
            for cluster in regularity_clusters:
                new_X[:, cluster] = np.round(new_X[:, cluster], self.precision)

        if new_X.shape[0] == 1:
            # converting a single array back to a 1D array
            new_X = new_X[0, :]

        return new_X

    def _find_precision(self, num):
        # function to find precision of a floating point number
        num_str = str(num)
        return max(len(num_str) - num_str.find(".") - 1, 0)

    def re_optimize(self):
        # use the same problem setting to run NSGA2 another time to handle cv
        new_problem_args = copy.deepcopy(self.problem_args)
        new_NSGA_settings = copy.deepcopy(self.NSGA_settings)

        # formulate the new shorter problem

        # dim of the new problem is the number of random independent and complete random variables
        new_problem_args["dim"] = len(self.rand_independent_vars) + len(self.rand_complete_vars)
        combined_rand_vars = self.rand_independent_vars + self.rand_complete_vars

        # take the corresponding lb and ubs
        new_problem_args["lb"] = self.lb if len(self.lb) == 1 else [
            self.lb[i] for i in combined_rand_vars]
        new_problem_args["ub"] = self.ub if len(self.ub) == 1 else [
            self.ub[i] for i in combined_rand_vars]

        # save the mapping of the variables of the smaller problem to the large problem
        new_problem_args["rand_variable_mapper"] = {
            "rand_independent_vars": list(np.arange(len(self.rand_independent_vars))),
            "rand_complete_vars": list(np.arange(len(self.rand_independent_vars), len(combined_rand_vars))),
        }

        # set the regularity information so that every time it enforces the regularity on its population
        new_problem_args["non_rand_vars"] = self.non_rand_vars
        new_problem_args["non_rand_vals"] = self.non_rand_vals
        new_problem_args["rand_cluster"] = self.rand_cluster
        new_problem_args["rand_dependent_vars"] = self.rand_dependent_vars
        new_problem_args["rand_dependent_lb"] = self.lb if len(self.lb) == 1 else [
            self.lb[i] for i in self.rand_dependent_vars]
        new_problem_args["rand_dependent_ub"] = self.ub if len(self.ub) == 1 else [
            self.ub[i] for i in self.rand_dependent_vars]
        new_problem_args["rand_independent_vars"] = self.rand_independent_vars
        new_problem_args["rand_complete_vars"] = self.rand_complete_vars
        new_problem_args["rand_complete_lb"] = self.lb if len(self.lb) == 1 else [
            self.lb[i] for i in self.rand_complete_vars]
        new_problem_args["rand_complete_ub"] = self.ub if len(self.ub) == 1 else [
            self.ub[i] for i in self.rand_complete_vars]

        # regularity_enforcement is True when we are dealing with the smaller problem
        new_problem_args["regularity_enforcement"] = True
        new_problem_args["regularity_enforcement_process"] = self._final_regularity_enforcement()

        # add constraints on the bounds of the dependent variables
        # following the regularitys may take some of them out of bounds
        new_problem_args["n_constr"] = self.problem_args["n_constr"] + (len(self.rand_dependent_vars) * 2)

        # get the smaller problem
        new_problem = get_problem(self.problem_name, new_problem_args)

        # before re-optimization, generate random samples for re-optimization
        # intermediate_init = Initialization(sampling=get_sampling("real_random"))
        # sample_X = intermediate_init.do(new_problem, self.NSGA_settings["pop_size"] * 10).get("X")
        # intermediate_X = np.zeros((sample_X.shape[0], self.problem_args["dim"]))
        # intermediate_X[:, self.rand_independent_vars] = sample_X
        # intermediate_X = self._final_regularity_enforcement()(intermediate_X)
        # intermediate_F = self.evaluate(intermediate_X, self.problem_args)
        # fronts = NonDominatedSorting().do(intermediate_F)
        # self.intermediate_X = intermediate_X[fronts[0], :]
        # self.intermediate_F = intermediate_F[fronts[0], :]

        # rerun NSGA2
        new_res = self.run_NSGA(new_problem, new_NSGA_settings)

        if new_res.X is not None:
            I = NonDominatedSorting().do(new_res.F)
            new_res.X = new_res.X[I[0], :]
            new_res.F = new_res.F[I[0], :]

            # get the solutions from the smaller problem and map them to the original problem
            new_X = np.zeros((new_res.X.shape[0], self.problem_args["dim"]))
            new_X[:, self.rand_independent_vars] = new_res.X[:, new_problem_args["rand_variable_mapper"][
                                                                    "rand_independent_vars"]]
            new_X[:, self.rand_complete_vars] = new_res.X[:, new_problem_args["rand_variable_mapper"][
                                                                    "rand_complete_vars"]]
            new_X = self._final_regularity_enforcement()(new_X)

            # set the X and evaluate the regularityed population
            self.X = new_X
            self.F = self.evaluate(self.X, self.problem_args)

        else:
            self.print("No feasible solution found with this regularity...")

    def _final_regularity_enforcement(self):
        # returns the function that enforces the final regularity to a given X
        def _regularity_enforcemnt(X):
            # enforce the final regularity to any vector of same size
            new_X = copy.deepcopy(X)

            # for non-random variables fix them to the found values
            if self.non_rand_cluster:
                new_X[:, self.non_rand_vars] = self.non_rand_vals

            # for random variables use the equation found in the process
            if self.rand_dependent_vars and self.rand_independent_vars:
                for i, dep_idx in enumerate(self.rand_dependent_vars):
                    new_X[:, dep_idx] = 0
                    for j, indep_idx in enumerate(self.rand_independent_vars):
                        new_X[:, dep_idx] += self.rand_final_reg_coef_list[i][j] * new_X[:, indep_idx]

                    new_X[:, dep_idx] += self.rand_final_reg_coef_list[i][-1]

            return new_X

        return _regularity_enforcemnt

    def _check_final_regularity(self):
        # create a population of random solutions and apply the regularity over them
        random_X = np.random.uniform(self.lb, self.ub, (self.X.shape[0]*10, self.problem_args["dim"]))
        regularityed_X = self.regularity.apply(random_X, self.lb, self.ub)
        regularityed_F, regularityed_G = self.evaluate(regularityed_X, self.problem_args, constr=True)

        if regularityed_G is not None:
            if len(regularityed_G.shape) == 1:
                regularityed_G = regularityed_G.reshape(-1, 1)
            constrained_idx = list(np.where(np.sum(regularityed_G > 0, axis=1) == 0)[0])
            regularityed_X = regularityed_X[constrained_idx, :]
            regularityed_F = regularityed_F[constrained_idx, :]

        required_pop_size = min(self.X.shape[0], regularityed_X.shape[0])
        regularityed_pop = pop_from_array_or_individual(regularityed_X)
        regularityed_pop.set("F", regularityed_F)

        if self.problem_args["n_obj"] == 2:
            # for 2-objective problems use rank and crowding survival
            survived_pop = RankAndCrowdingSurvival()._do(self.problem, regularityed_pop, n_survive=required_pop_size)
            self.X = survived_pop.get("X")
            self.F = survived_pop.get("F")

        elif self.problem_args["n_obj"] > 2:
            # for >2 objectives use reference direction based survival
            survived_pop = ReferenceDirectionSurvival(self.NSGA_settings["ref_dirs"])._do(self.problem, regularityed_pop, n_survive=required_pop_size)
            self.X = survived_pop.get("X")
            self.F = survived_pop.get("F")

        else:
            print("[Error!] Wrong dimensions")




