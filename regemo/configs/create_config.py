from regemo.utils.path_utils import create_dir
import regemo.config as config
from regemo.problems.get_problem import problems


import os
import sys
import pickle
import numpy as np

from pymoo.factory import get_reference_directions

# problems = ["car_side_impact", "conceptual_marine_design", "rocket_injector_design", "dtlz5"]
problems = ["bnh"]


def create_config(problem_name,
                  non_fixed_regularity_coef_factor=0.5,
                  non_fixed_dependency_percent=0.5,
                  delta=0.05,
                  n_rand_bins=3,
                  non_fixed_regularity_degree=1):

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
            algorithm_config["non_fixed_regularity_coef_factor"] = non_fixed_regularity_coef_factor
            algorithm_config["non_fixed_dependency_percent"] = non_fixed_dependency_percent
            algorithm_config["delta"] = delta
            algorithm_config["n_rand_bins"] = n_rand_bins
            algorithm_config["non_fixed_regularity_degree"] = non_fixed_regularity_degree
            problem_config["visualization_angle"] = (34, 29)
    else:

        problem_config = {
            "name": problem_name,
            "dim": 10,
            "n_obj": 2,
            "n_constr": 0,
            "lb": [0]*10,
            "ub": [1]*10,
            "visualization_angle": (45, 45),
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
        precision = 2

        algorithm_config["NSGA_settings"] = NSGA_settings
        algorithm_config["clustering_config"] = algorithm_clustering_config
        algorithm_config["precision"] = precision
        algorithm_config["non_fixed_regularity_coef_factor"] = non_fixed_regularity_coef_factor
        algorithm_config["non_fixed_dependency_percent"] = non_fixed_dependency_percent
        algorithm_config["delta"] = delta
        algorithm_config["n_rand_bins"] = n_rand_bins
        algorithm_config["non_fixed_regularity_degree"] = non_fixed_regularity_degree

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