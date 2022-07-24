from regemo.utils.path_utils import create_dir
import regemo.config as config

import os
import sys
import pickle

problems = ["bnh", "c2dtlz2", "crashworthiness", "dtlz2", "dtlz5", "dtlz7", "mod_zdt", "osy", "scalable_truss", "srn",
            "two_member_truss", "welded_beam_design"]
# problems = ["bnh"]

def create_config(problem_name):
    non_rand_regularity_degree = 1
    rand_regularity_coef_factor = 0.1
    rand_regularity_dependency = 1
    rand_factor_sd = 0.3
    precision = 2
    rand_regularity_MSE_threshold = 0.1
    non_rand_regularity_MSE_threshold = 0.3
    cluster_pf_required = True
    pf_cluster_eps = None
    n_clusters = 1

    NSGA_settings = {}
    NSGA_settings["pop_size"] = 200
    NSGA_settings["n_offsprings"] = 10
    NSGA_settings["mut_eta"] = 50
    NSGA_settings["sbx_eta"] = 20

    use_existing_config = True
    save_config = True

    algorithm_config_storage_dir = config.algorithm_config_path
    problem_config_storage_dir = config.problem_config_path
    algorithm_config = {}

    # create the dirs for storing config files
    if not os.path.exists(algorithm_config_storage_dir):
        create_dir(algorithm_config_storage_dir)
    if not os.path.exists(problem_config_storage_dir):
        create_dir(problem_config_storage_dir)

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
            algorithm_config["n_clusters"] = n_clusters
            
            # algorithm_config["non_rand_regularity_degree"] = non_rand_regularity_degree
            # algorithm_config["rand_regularity_coef_factor"] = rand_regularity_coef_factor
            # algorithm_config["rand_regularity_dependency"] = rand_regularity_dependency
            # algorithm_config["rand_factor_sd"] = rand_factor_sd
            # algorithm_config["precision"] = precision
            # algorithm_config["rand_regularity_MSE_threshold"] = rand_regularity_MSE_threshold
            # algorithm_config["non_rand_regularity_MSE_threshold"] = non_rand_regularity_MSE_threshold
            # algorithm_config["cluster_pf_required"] = cluster_pf_required
            # algorithm_config["pf_cluster_eps"] = pf_cluster_eps
            #
            # algorithm_config["NSGA_settings"]["pop_size"] = NSGA_settings["pop_size"]
            # algorithm_config["NSGA_settings"]["n_offsprings"] = NSGA_settings["n_offsprings"]
            # algorithm_config["NSGA_settings"]["mut_eta"] = NSGA_settings["mut_eta"]
            # algorithm_config["NSGA_settings"]["sbx_eta"] = NSGA_settings["sbx_eta"]
            #
            # problem_config["clustering_config"] = {"criterion": "X"}
            # problem_config["n_clusters"] = 3
            # problem_config["n_constr"] = 4
            # problem_config["n_constr"] = 2
    else:
        # HDBSCAN parameters
        problem_clustering_config = {
            "criterion": "F",
            "hdbscan_args": {
                "min_cluster_size": 2,
                "min_samples": 3,
                "cluster_selection_epsilon": 0,
            }
        }

        problem_config = {
            "name": problem_name,
            "dim": 279,
            "n_obj": 2,
            "n_constr": 2,
            "lb": [0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,
                   0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5],
            "ub": [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,29.0,29.0,29.0,29.0,29.0,29.0,29.0,29.0,29.0,29.0,29.0,29.0,29.0,29.0,29.0,29.0,29.0,29.0,29.0],
            "visualization_angle": (0, 0),
            "clustering_config": problem_clustering_config
        }

        NSGA_settings = {"pop_size": 100,
                         "n_offsprings": 30,
                         "sbx_prob": 0.9,
                         "sbx_eta": 3,
                         "mut_eta": 20,
                         "n_eval": 40000}

        #  for 3 or more objectives
        # NSGA_settings["ref_dirs"] = get_reference_directions("das-dennis", 3, n_partitions=12)

        algorithm_clustering_config = {
            "min_cluster_size": 2,
            "max_clusters": 4,
            "MSE_threshold": 0.0002
        }

        non_rand_regularity_degree = 2  # [1, 2]
        rand_regularity_coef_factor = 0.001  # [0.1 - 0.5]
        rand_regularity_dependency = 1  # [1, 2]
        precision = 2  # [0, 1, 2, 3]
        rand_factor_sd = 0.1
        rand_regularity_MSE_threshold = 0.5

        algorithm_config["NSGA_settings"] = NSGA_settings
        algorithm_config["clustering_config"] = algorithm_clustering_config
        algorithm_config["non_rand_regularity_degree"] = non_rand_regularity_degree
        algorithm_config["rand_regularity_coef_factor"] = rand_regularity_coef_factor
        algorithm_config["rand_regularity_dependency"] = rand_regularity_dependency
        algorithm_config["precision"] = precision
        algorithm_config["rand_factor_sd"] = rand_factor_sd
        algorithm_config["rand_regularity_MSE_threshold"] = rand_regularity_MSE_threshold
        algorithm_config["cluster_pf_required"] = False

    # store the algorithm and problem configurations
    if save_config:
        with open(f"{algorithm_config_storage_dir}/{problem_name}.pickle", "wb") as pkl_handler:
            pickle.dump(algorithm_config, pkl_handler)
            pkl_handler.close()

        with open(f"{problem_config_storage_dir}/{problem_name}.pickle", "wb") as pkl_handler:
            pickle.dump(problem_config, pkl_handler)
            pkl_handler.close()

    print(f"Final problem config: {problem_config}")
    print(f"Final algorithm config: {algorithm_config}")

if __name__ == "__main__":
    for problem in problems:
        create_config(problem)