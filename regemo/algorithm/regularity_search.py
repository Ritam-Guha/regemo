from regemo.problems.get_problem import problems as problem_set
from regemo.problems.get_problem import get_problem
from regemo.algorithm.regularity_finder import Regularity_Finder
from regemo.utils.path_utils import create_dir
import regemo.config as config
from regemo.algorithm.nsga3 import NSGA3

from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import HyperplaneNormalization
from pymoo.algorithms.moo.nsga3 import associate_to_niches
from pymoo.visualization.scatter import Scatter
from pymoo.factory import get_sampling, get_crossover, get_mutation, get_termination, get_performance_indicator
from pymoo.optimize import minimize
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

import inspect
import argparse
import copy
import os
import sys
from hdbscan.hdbscan_ import HDBSCAN
from pymoo.visualization.pcp import PCP

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pickle

plt.rcParams.update({'font.size': 10})

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
                 non_fixed_dependency_percent=0.9,
                 non_fixed_regularity_coef_factor=0.1,
                 non_fixed_regularity_degree=1,
                 delta=0.05,
                 precision=2,
                 n_rand_bins=20,
                 NSGA_settings=None,
                 save_img=True,
                 result_storage=None,
                 verbose=True,
                 seed=0):
        """
        :param problem_args: parameters for the problem
        :param non_fixed_dependency_percent: percentage of non-fixed variables to be treated as non-fixed dependent
        :param non_fixed_regularity_coef_factor: coefficients should be a multiple of this factor
        :param delta: threshold for identifying fixed variables
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
        self.regularity_objs = None
        self.problem_name = problem_args["name"]
        self.problem_args = problem_args
        self.seed = seed
        self.non_fixed_dependency_percent = non_fixed_dependency_percent
        self.non_fixed_regularity_coef_factor = non_fixed_regularity_coef_factor
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
        self.save_img = save_img
        self.verbose = verbose
        self.print = self.verboseprint()
        self.visualization_angle = self.problem_args["visualization_angle"]
        self.ideal_point = None
        self.nadir_point = None

        # final objectives
        self.hv = None
        self.final_metrics = {
            "complexity": 0,
            "hv_dif_%": 0
        }

        np.random.seed(self.seed)

    def run(self):
        # the main running function
        initial_pop_storage = f"{config.BASE_PATH}/results/hierarchical_search/{self.problem_name}/" \
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

        # plot the figure after nds
        plot = Scatter(labels="F", legend=True, angle=self.visualization_angle)
        plot = plot.add(res["F"], color="blue", label="Original PO Front", alpha=0.2, s=60)

        if self.save_img:
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
                                                   non_fixed_dependency_percent=self.non_fixed_dependency_percent,
                                                   non_fixed_regularity_coef_factor=self.non_fixed_regularity_coef_factor,
                                                   non_fixed_regularity_degree=self.non_fixed_regularity_degree,
                                                   precision=self.precision,
                                                   NSGA_settings=new_NSGA_settings,
                                                   seed=self.seed,
                                                   save_img=self.save_img,
                                                   result_storage=self.result_storage,
                                                   verbose=self.verbose,
                                                   n_rand_bins=self.n_rand_bins,
                                                   delta=self.delta)

        regularity_enforcement.run()    # run the regularity enforcement
        self.final_metrics["complexity"] += regularity_enforcement.regularity.calc_process_complexity()

        # storage file for every PF
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
        self.print("Final Metrics")
        self.print(f"IGD+: {regularity_enforcement.final_metrics['igd_plus']}")
        self.print(f"HV_dif_%: {regularity_enforcement.final_metrics['hv_dif_%']}")
        self.print("\n======================================\n")


        # get the X and F after regularity enforcement
        self.regular_X = regularity_enforcement.X
        self.regular_F = regularity_enforcement.F

        # save the objects
        self.regularity_obj = regularity_enforcement

        # plot the regular front
        if self.regular_F is not None:
            plot = Scatter(labels="F", legend=True, angle=self.visualization_angle)
            plot = plot.add(self.regular_F, color="red", marker="*", s=60, alpha=0.6, label="Regular Efficient Front")

            if self.verbose and plot:
                plot.show()

            if self.save_img and plot:
                plot.save(f"{config.BASE_PATH}/{self.result_storage}/regular_efficient_front.pdf", format="pdf")

        # plot the original and regular front
        plot = Scatter(labels="F", legend=True, angle=self.visualization_angle, tight_layout=True, fontsize=10)
        plot = plot.add(self.orig_F, color="blue", marker="o", s=60, alpha=0.2, label="Original PO Front")
        if self.regular_F is not None:
            plot = plot.add(self.regular_F, color="red", marker="*", s=50, alpha=0.6, label="Regular Front")

        if self.verbose:
            plot.show()

        if self.save_img:
            plot.save(
                f"{config.BASE_PATH}/{self.result_storage}/final_efficient_fronts.pdf", format="pdf", dpi=200)

        # calculate the HV_diff_%
        orig_hv = regularity_enforcement.orig_hv
        new_hv = regularity_enforcement.regularity_hv

        if new_hv > 0:
            self.final_metrics["hv_dif_%"] = ((abs(orig_hv - new_hv)) / orig_hv) * 100
        else:
            # when it converges to 1 point
            self.final_metrics["hv_dif_%"] = np.inf

        self.print(f"Overall complexity: {self.final_metrics['complexity']}, HV_diff_%: "
                   f"{self.final_metrics['hv_dif_%']}")

        # store the initial population
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

            if self.save_img:
                plot.save(f"{config.BASE_PATH}/{self.result_storage}/PCP_final_population.pdf", format="pdf")

            if self.verbose:
                plot.show()

        with open(f"{config.BASE_PATH}/{self.result_storage}/final_metrics.txt", "w") as f:
            f.write(f"HV_dif_%: {self.final_metrics['hv_dif_%']}\n")
            f.write(f"complexity: {self.final_metrics['complexity']}")
        print(f"hv_diff_%; {self.final_metrics['hv_dif_%']}")
        print(f"complexity: {self.final_metrics['complexity']}")

    def run_NSGA(self, problem, NSGA_settings):
        # run the NSGA over the problem
        if self.problem_args["n_obj"] == 2:
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
            self.print("Running NSGA3..")
            algorithm = NSGA3(pop_size=NSGA_settings["pop_size"],
                              n_offsprings=NSGA_settings["n_offsprings"],
                              sampling=get_sampling("real_random"),
                              crossover=get_crossover("real_sbx", prob=NSGA_settings["sbx_prob"],
                                                      eta=NSGA_settings["sbx_eta"]),
                              mutation=get_mutation("real_pm", eta=NSGA_settings["mut_eta"]),
                              ref_dirs=NSGA_settings["ref_dirs"],
                              seed=self.seed,
                              eliminate_duplicate=True,
                              verbose=self.verbose)

        else:
            print("[INFO] Not suitable for less than 2 objectives...")
            sys.exit(1)

        termination = get_termination("n_eval", NSGA_settings["n_eval"])

        res = minimize(problem,
                       algorithm,
                       termination,
                       seed=self.seed,
                       verbose=True,
                       save_history=True)

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
        niche_of_individuals, dist_to_niche, dist_matrix = associate_to_niches(F, ref_dirs, self.ideal_point,
                                                                               self.nadir_point)
        closest_ref_dir_index = np.argmin(dist_matrix, axis=1)

        closest_ref_dirs = ref_dirs[closest_ref_dir_index, :]
        return closest_ref_dirs

    def k_means_cluster(self, pf_res, n_clusters=3, clustering_criterion="X"):
        # function to cluster the pareto front solutions
        # initialize the clusters
        clusters = []

        if clustering_criterion == "X":
            pf = pf_res["X"]
        else:
            pf = pf_res["F"]

        # normalize the embedded pf
        norm_pf = normalize(pf, axis=0, norm="max")

        # get the labels from k-means
        labels = KMeans(n_clusters=n_clusters, random_state=0, n_init=30).fit_predict(pf)
        # put all the outliers to a different cluster
        n_clusters = len(set(labels))
        n_clusters = n_clusters - 1 if -1 in labels else n_clusters

        # plot the clustering output
        colors = cm.rainbow(np.linspace(0, 1, n_clusters))
        fig = plt.figure(figsize=(8, 6))

        # define the plotting axes
        pf_F = pf_res["F"]

        if pf_F.shape[1] > 3:
            pca = PCA(n_components=2)
            embedded_pf_F = pca.fit_transform(pf_F)
            ax = plt.axes()

        elif pf_F.shape[1] == 3:
            ax = plt.axes(projection="3d")

        elif pf_F.shape[1] == 2:
            ax = plt.axes()

        else:
            ax = None

        if ax:
            # if self.visualization_angle is not None:
            #     ax.view_init(*self.visualization_angle)

            # generate a scatter plot for the clusters
            for c, i in zip(colors, set(labels)):
                if i != -1:
                    if pf_F.shape[1] == 3:
                        ax.scatter3D(pf_res["F"][labels == i, 0], pf_res["F"][labels == i, 1],
                                     pf_res["F"][labels == i, 2],
                                     color=c,
                                     label="Cluster " + str(i + 1))

                    elif pf_F.shape[1] == 2:
                        ax.scatter(pf_res["F"][labels == i, 0], pf_res["F"][labels == i, 1], color=c,
                                   label="Cluster " + str(i + 1))

            plt.legend()
            # plt.title(f"Clustering View on F Space")

        else:
            print("Does not work for less than 2 objectives")
            plt.show()

        # append the clusters one after another
        for i in range(n_clusters):
            cur_cluster_X = pf_res["X"][labels == i, :]
            cur_cluster_F = pf_res["F"][labels == i, :]

            clusters.append({"X": cur_cluster_X, "F": cur_cluster_F})

        if self.save_img:
            fig.savefig(f"{config.BASE_PATH}/{self.result_storage}/clustering_efficient_front.pdf", format="pdf")

        if self.verbose:
            plt.show()

        dim = pf_res["X"].shape[1]
        fig = plt.figure(figsize=(8, 6))
        for c, i in zip(colors, set(labels)):
            if i != -1:
                for j in range(pf_res["X"].shape[0]):
                    if labels[j] == i:
                        plt.scatter(np.arange(dim), pf_res["X"][j, :], color=c)

        plt.xticks(np.arange(dim), labels=[f"$X_{i + 1}$" for i in range(dim)])
        plt.xlabel("Variables")
        # plt.title(f"Clustering View on X Space")

        plt.tick_params(axis="x", labelsize=10, labelrotation=40)
        plt.tick_params(axis="y", labelsize=10, labelrotation=20)

        if self.save_img:
            fig.savefig(f"{config.BASE_PATH}/{self.result_storage}/clustering_pareto_front.pdf", format="pdf")

        if self.verbose:
            plt.show()

        return clusters

    def cluster_pf(self, pf_res, user_hdbscan_args, clustering_criterion="X"):
        # function to cluster the pareto front solutions

        # initialize the clusters
        clusters = []

        if clustering_criterion == "X":
            pf = pf_res["X"]
        else:
            pf = pf_res["F"]

        # normalize the embedded pf
        norm_pf = normalize(pf, axis=0, norm="max")

        # get the default parameters of HDBSCAN and populate it with user_defined parameters
        dict_hdbscan_args = get_default_args(HDBSCAN)
        for k in user_hdbscan_args.keys():
            dict_hdbscan_args[k] = user_hdbscan_args[k]

        # get the labels from hdbscan
        labels = HDBSCAN(min_cluster_size=dict_hdbscan_args["min_cluster_size"], min_samples=dict_hdbscan_args[
            "min_samples"], cluster_selection_epsilon=dict_hdbscan_args["cluster_selection_epsilon"]).fit_predict(
            norm_pf)

        # put all the outliers to a different cluster
        n_clusters = len(set(labels))
        n_clusters = n_clusters - 1 if -1 in labels else n_clusters

        # plot the clustering output
        colors = cm.rainbow(np.linspace(0, 1, n_clusters))
        fig = plt.figure(figsize=(15, 8))

        # define the plotting axes
        pf_F = pf_res["F"]

        if pf_F.shape[1] > 3:
            pca = PCA(n_components=2)
            embedded_pf_F = pca.fit_transform(pf_F)
            ax = plt.axes()

        elif pf_F.shape[1] == 3:
            ax = plt.axes(projection="3d")

        elif pf_F.shape[1] == 2:
            ax = plt.axes()

        else:
            ax = None

        if ax:
            # if self.visualization_angle is not None:
            #     ax.view_init(*self.visualization_angle)

            # generate a scatter plot for the clusters
            for c, i in zip(colors, set(labels)):
                if i != -1:
                    if pf_F.shape[1] == 3:
                        ax.scatter3D(pf_res["F"][labels == i, 0], pf_res["F"][labels == i, 1],
                                     pf_res["F"][labels == i, 2],
                                     color=c,
                                     label="Cluster " + str(i + 1))

                    elif pf_F.shape[1] == 2:
                        ax.scatter(pf_res["F"][labels == i, 0], pf_res["F"][labels == i, 1], color=c,
                                   label="Cluster " + str(i + 1))

            plt.legend()
            # plt.title(f"Clustering on {clustering_criterion} space")

        else:
            print("Does not work for less than 2 objectives")
            plt.show()

        # append the clusters one after another
        for i in range(n_clusters):
            cur_cluster_X = pf_res["X"][labels == i, :]
            cur_cluster_F = pf_res["F"][labels == i, :]

            clusters.append({"X": cur_cluster_X, "F": cur_cluster_F})

        if self.save_img:
            fig.savefig(f"{config.BASE_PATH}/{self.result_storage}/efficient_front_clustering.pdf", format="pdf")

        if self.verbose:
            plt.show()

        return clusters

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


def main(problem_name="crashworthiness"):
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
                                              non_fixed_regularity_coef_factor=algorithm_config[
                                                  "non_fixed_regularity_coef_factor"],
                                              precision=algorithm_config["precision"],
                                              n_rand_bins=algorithm_config["n_rand_bins"],
                                              delta=algorithm_config["delta"],
                                              non_fixed_dependency_percent=algorithm_config[
                                                  "non_fixed_dependency_percent"],
                                              non_fixed_regularity_degree=algorithm_config[
                                                  "non_fixed_regularity_degree"],
                                              save_img=True,
                                              result_storage=f"{res_storage_dir}",
                                              verbose=False)

        # run the search object
        regularity_search.run()


if __name__ == "__main__":
    main()
