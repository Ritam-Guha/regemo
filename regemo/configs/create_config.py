import numpy as np

from regemo.utils.path_utils import create_dir
import regemo.config as config
from regemo.problems.get_problem import problems

import os
import sys
import pickle
from pymoo.factory import get_reference_directions

problems = ["rocket_injector_design"]

def create_config(problem_name):
    # non_rand_regularity_degree = 1
    # rand_regularity_coef_factor = 0.1
    # rand_regularity_dependency = 1
    # rand_factor_sd = 0.2
    # precision = 2
    # rand_regularity_MSE_threshold = 0.1
    # non_rand_regularity_MSE_threshold = 0.1
    # cluster_pf_required = True
    # n_clusters = 1
    # visualization_angle = (45, 45)
    # n_obj = 3
    # n_constr = 9

    # NSGA_settings = {}
    # NSGA_settings["pop_size"] = 1000
    # NSGA_settings["n_offsprings"] = 100
    # NSGA_settings["mut_eta"] = 50
    # NSGA_settings["sbx_eta"] = 20
    pop_size = 1000

    use_existing_config = True
    save_config = True

    algorithm_config_storage_dir = config.algorithm_config_path
    problem_config_storage_dir = config.problem_config_path
    algorithm_config = {}

    # create the dirs for storing config files
    if not os.path.exists(f"{config.BASE_PATH}/{algorithm_config_storage_dir}"):
        create_dir(algorithm_config_storage_dir)
    if not os.path.exists(f"{config.BASE_PATH}/{problem_config_storage_dir}"):
        create_dir(problem_config_storage_dir)

    if use_existing_config:
        if not os.path.exists(f"{config.BASE_PATH}/{problem_config_storage_dir}/{problem_name}.pickle"):
            print("[Error!] Problem Configuration file not found...")
            sys.exit(1)
        if not os.path.exists(f"{config.BASE_PATH}/{algorithm_config_storage_dir}/{problem_name}.pickle"):
            print("[Error!] Algorithm Configuration file not found...")
            sys.exit(1)
        else:
            problem_config = pickle.load(open(f"{config.BASE_PATH}/{problem_config_storage_dir}/{problem_name}.pickle", "rb"))
            algorithm_config = pickle.load(open(f"{config.BASE_PATH}/{algorithm_config_storage_dir}/{problem_name}.pickle", "rb"))
            # algorithm_config["non_rand_regularity_degree"] = non_rand_regularity_degree
            # algorithm_config["rand_regularity_coef_factor"] = rand_regularity_coef_factor
            # algorithm_config["rand_regularity_dependency"] = rand_regularity_dependency
            # algorithm_config["rand_factor_sd"] = rand_factor_sd
            # algorithm_config["precision"] = precision
            # algorithm_config["rand_regularity_MSE_threshold"] = rand_regularity_MSE_threshold
            # algorithm_config["non_rand_regularity_MSE_threshold"] = non_rand_regularity_MSE_threshold
            # algorithm_config["cluster_pf_required"] = cluster_pf_required
            # algorithm_config["pf_cluster_eps"] = pf_cluster_eps
            # problem_config["name"] = problem_name
            # if problem_config["n_obj"] == 3:
            #     problem_config["visualization_angle"] = (45, 45)
            # problem_config["n_obj"] = n_obj
            # problem_config["n_constr"] = n_constr

            # for key in list(algorithm_config.keys()):
            #     if "pattern" in key:
            #         del algorithm_config[key]
            #
            # algorithm_config["n_clusters"] = n_clusters
            # algorithm_config["non_rand_regularity_degree"] = non_rand_regularity_degree
            # algorithm_config["rand_regularity_coef_factor"] = rand_regularity_coef_factor
            # algorithm_config["rand_regularity_dependency"] = rand_regularity_dependency
            # algorithm_config["rand_factor_sd"] = rand_factor_sd
            # algorithm_config["precision"] = precision
            # algorithm_config["rand_regularity_MSE_threshold"] = rand_regularity_MSE_threshold
            # algorithm_config["non_rand_regularity_MSE_threshold"] = non_rand_regularity_MSE_threshold
            # algorithm_config["cluster_pf_required"] = cluster_pf_required
            #
            algorithm_config["NSGA_settings"]["pop_size"] = pop_size
            # algorithm_config["NSGA_settings"]["n_offsprings"] = NSGA_settings["n_offsprings"]
            # algorithm_config["NSGA_settings"]["mut_eta"] = NSGA_settings["mut_eta"]
            # algorithm_config["NSGA_settings"]["sbx_eta"] = NSGA_settings["sbx_eta"]
            # if problem_config["n_obj"] > 2:
            #     algorithm_config["NSGA_settings"]["ref_dirs"] = get_reference_directions("das-dennis", problem_config["n_obj"], n_partitions=12)

            # problem_config["clustering_config"] = {"criterion": "X"}
            # problem_config["n_clusters"] = 3
            # problem_config["n_constr"] = 4
            # problem_config["n_constr"] = 2
    else:
        # HDBSCAN parameters
        # problem_clustering_config = {
        #     "criterion": "F",
        #     "hdbscan_args": {
        #         "min_cluster_size": 2,
        #         "min_samples": 3,
        #         "cluster_selection_epsilon": 0,
        #     }
        # }

        problem_config = {
            "name": problem_name,
            "dim": 7,
            "n_obj": 9,
            "n_constr": 0,
            "lb": [0.5, 0.45, 0.5, 0.5, 0.875, 0.4, 0.4],
            "ub": [1.5, 1.35, 1.5, 1.5, 2.625, 1.2, 1.2],
            "visualization_angle": (0, 0),
        }

        NSGA_settings = {"pop_size": 200, "n_offsprings": 30, "sbx_prob": 1, "sbx_eta": 20, "mut_eta": 20,
                         "n_eval": 40000,
                         "ref_dirs": get_reference_directions("das-dennis", problem_config["n_obj"], n_partitions=12)}

        #  for 3 or more objectives

        algorithm_clustering_config = {
            "min_cluster_size": 2,
            "max_clusters": 4,
            "MSE_threshold": 0.0002
        }

        non_rand_regularity_degree = 2  # [1, 2]
        rand_regularity_coef_factor = 0.5  # [0.1 - 0.5]
        rand_regularity_dependency = 1  # [1, 2]
        precision = 2  # [0, 1, 2, 3]
        rand_factor_sd = 0.1
        rand_regularity_MSE_threshold = 0.5
        non_rand_regularity_MSE_threshold = 0.3
        n_clusters = 1

        algorithm_config["NSGA_settings"] = NSGA_settings
        algorithm_config["clustering_config"] = algorithm_clustering_config
        algorithm_config["non_rand_regularity_degree"] = non_rand_regularity_degree
        algorithm_config["rand_regularity_coef_factor"] = rand_regularity_coef_factor
        algorithm_config["rand_regularity_dependency"] = rand_regularity_dependency
        algorithm_config["precision"] = precision
        algorithm_config["rand_factor_sd"] = rand_factor_sd
        algorithm_config["rand_regularity_MSE_threshold"] = rand_regularity_MSE_threshold
        algorithm_config["non_rand_regularity_MSE_threshold"] = non_rand_regularity_MSE_threshold
        algorithm_config["cluster_pf_required"] = False
        algorithm_config["n_clusters"] = n_clusters

    # store the algorithm and problem configurations
    if save_config:
        with open(f"{config.BASE_PATH}/{algorithm_config_storage_dir}/{problem_name}.pickle", "wb") as pkl_handler:
            pickle.dump(algorithm_config, pkl_handler)
            pkl_handler.close()

        with open(f"{config.BASE_PATH}/{problem_config_storage_dir}/{problem_name}.pickle", "wb") as pkl_handler:
            pickle.dump(problem_config, pkl_handler)
            pkl_handler.close()

    print(f"Final problem config: {problem_config}")
    print(f"Final algorithm config: {algorithm_config}")


if __name__ == "__main__":
    for problem in problems:
        create_config(problem)