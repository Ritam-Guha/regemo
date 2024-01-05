import numpy as np

from regemo.utils.path_utils import create_dir
import regemo.config as config
from regemo.problems.get_problem import problems

import os
import sys
import pickle
from pymoo.util.ref_dirs import get_reference_directions

# problems = ["car_side_impact", "conceptual_marine_design", "rocket_injector_design", "dtlz5"]
problems = ["crashworthiness"]


def create_config(problem_name,
                  non_fixed_regularity_coef_factor=0.1,
                  non_fixed_dependency_percent=0.5,
                  delta=0.05,
                  n_rand_bins=5,
                  non_fixed_regularity_degree=1,
                  **kwargs):

    use_existing_config = False
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
        if "scalable_truss" in problem_name:
            problem_config = {"name": problem_name, "dim": kwargs["n_size_var"] + kwargs["n_shape_var"], "n_obj": 2,
                              "n_constr": 2, "visualization_angle": (45, 45)}

            if kwargs["shape_var_mode"] == 'z':
                # Represent truss shape by varying the z coordinates
                xl = np.concatenate((0.005 * np.ones(kwargs["n_size_var"]), kwargs["zmin"] * np.ones(
                    kwargs["n_shape_var"])))
                xu = np.concatenate((0.100 * np.ones(kwargs["n_size_var"]), kwargs["zmax"] * np.ones(
                    kwargs["n_shape_var"])))
            else:
                # Represent truss shape by through the length (l) of the vertical members. l = 4 - z-coord
                xl = np.concatenate((0.005 * np.ones(kwargs["n_size_var"]), kwargs["lmin"] * np.ones(kwargs[
                                                                                                      "n_shape_var"])))
                xu = np.concatenate((0.100 * np.ones(kwargs["n_size_var"]), kwargs["lmax"] * np.ones(kwargs[
                                                                                                         "n_shape_var"])))
            problem_config["lb"] = xl
            problem_config["ub"] = xu

        elif "solid_rocket" in problem_name:
            problem_config = {"name": problem_name, "dim": 227, "n_obj": 2,
                              "n_constr": 0, "visualization_angle": (45, 45)}

            xl = pickle.load(open(f"{config.BASE_PATH}/problems/solid_rocket_design/xl.pickle", "rb"))
            xu = pickle.load(open(f"{config.BASE_PATH}/problems/solid_rocket_design/xu.pickle", "rb"))

            problem_config["lb"] = xl
            problem_config["ub"] = xu

        else:
            problem_config = {"name": problem_name, "dim": 3, "n_obj": 4,
                              "n_constr": 3, "lb": [6.4, 0.69, 3.91], "ub": [7.09, 2.89, 4.61], "visualization_angle":
                                  (45, 45)}

        NSGA_settings = {"pop_size": 100, "n_offsprings": 30, "sbx_prob": 1, "sbx_eta": 20, "mut_eta": 20,
                         "n_eval": 40000}
        if problem_config["n_obj"] > 2:
            NSGA_settings["ref_dirs"] = get_reference_directions("das-dennis", problem_config["n_obj"], n_partitions=12)

        algorithm_config["NSGA_settings"] = NSGA_settings
        algorithm_config["precision"] = 3
        algorithm_config["n_rand_bins"] = 5
        algorithm_config["delta"] = 0.5
        algorithm_config["non_fixed_regularity_degree"] = 2
        algorithm_config["num_non_fixed_independent_vars"] = 1

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
    # create_config(problem_name="scalable_truss_59",
    #               n_size_var=820,
    #               n_shape_var=59,
    #               shape_var_mode="l",
    #               lmin=0.5,
    #               lmax=29)
    create_config(problem_name="solid_rocket_design")
    # for problem in problems:
    #     create_config(problem)