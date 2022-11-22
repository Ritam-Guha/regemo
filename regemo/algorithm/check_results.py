import regemo.config as config
from regemo.problems.get_problem import problems

import pickle
import os


def complete_check(problem_name):
    _check_ranges(problem_name=problem_name)


def _check_ranges(problem_name):
    folder_path = f"{config.BASE_PATH}/results/{problem_name}/"
    list_files = os.listdir(folder_path)
    list_files = [file for file in list_files if "regularity_principle_" in file]

    try:
        for file in list_files:
            regularity_principle = pickle.load(open(f"{config.BASE_PATH}/results/{problem_name}/{file}", "rb"))
            # check non rand variables
            for i, var in enumerate(regularity_principle["non_rand_vars"]):
                assert(regularity_principle["problem_config"]["lb"][var] <=
                       regularity_principle["non_rand_vals"][i] <=
                       regularity_principle["problem_config"]["ub"][var])

            # check rand orphan variables
            for i, var in enumerate(regularity_principle["rand_orphan_vars"]):
                assert(regularity_principle["problem_config"]["lb"][var] <=
                       regularity_principle["lb"][var] and
                       regularity_principle["problem_config"]["ub"][var] >=
                       regularity_principle["ub"][var])

            # check rand independent variables
            for i, var in enumerate(regularity_principle["rand_independent_vars"]):
                assert(regularity_principle["problem_config"]["lb"][var] <=
                       regularity_principle["lb"][var] and
                       regularity_principle["problem_config"]["ub"][var] >=
                       regularity_principle["ub"][var])

            # check rand dependent variables
            for i, var in enumerate(regularity_principle["rand_dependent_vars"]):
                cur_lb = 0
                cur_ub = 0
                for j, indep_var in enumerate(regularity_principle["rand_independent_vars"]):
                    if regularity_principle["coef_list"][i, j] > 0:
                        cur_lb += regularity_principle["coef_list"][i, j] * regularity_principle["lb"][indep_var]
                        cur_ub += regularity_principle["coef_list"][i, j] * regularity_principle["ub"][indep_var]

                    elif regularity_principle["coef_list"][i, j] < 0:
                        cur_lb += regularity_principle["coef_list"][i, j] * regularity_principle["ub"][indep_var]
                        cur_ub += regularity_principle["coef_list"][i, j] * regularity_principle["lb"][indep_var]

                cur_lb += regularity_principle["coef_list"][i, -1]
                cur_ub += regularity_principle["coef_list"][i, -1]

                assert(regularity_principle["problem_config"]["lb"][var] <=
                       cur_lb and
                       regularity_principle["problem_config"]["ub"][var] >=
                       cur_ub)

            # check if number of random variables satisfy the geometry
            assert(len(regularity_principle["rand_orphan_vars"])+len(regularity_principle["rand_independent_vars"])
                   <= regularity_principle["problem_config"]["n_obj"]-1)

            # assert check if it's not a point
            assert(len(regularity_principle["rand_orphan_vars"]) +
                   len(regularity_principle["rand_dependent_vars"]) +
                   len(regularity_principle["rand_independent_vars"]) > 0)

    except AssertionError:
        print(f"{problem_name} does not satisfy requirement")
        return None


def main():
    for problem_name in problems:
        complete_check(problem_name=problem_name)


if __name__ == "__main__":
    main()