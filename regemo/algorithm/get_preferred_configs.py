import regemo.config as config
from regemo.problems.get_problem import problems

import pickle


def find_preferred_id(cur_problems):
    for problem in cur_problems:
        try:
            cur_id = int(pickle.load(
                open(f"{config.BASE_PATH}/results/hierarchical_search/{problem}/preferred_id.pickle", "rb")))
            print(f"{problem}: {cur_id}")

        except:
            print(f"preferred id not found for {problem}")
            continue

        param = pickle.load(
            open(f"{config.BASE_PATH}/results/hierarchical_search/{problem}/param_comb_{cur_id}/param_comb.pkl", "rb"))
        problem_config = pickle.load(open(f"{config.BASE_PATH}/configs/problem_configs/{problem}.pickle", "rb"))
        algorithm_config = pickle.load(open(f"{config.BASE_PATH}/configs/algorithm_configs/{problem}.pickle", "rb"))

        common_problem_config = list(set(problem_config.keys()).intersection(set(param.keys())))
        common_algorithm_config = list(set(algorithm_config.keys()).intersection(set(param.keys())))

        for key in common_problem_config:
            problem_config[key] = param[key]
        for key in common_algorithm_config:
            algorithm_config[key] = param[key]

        for key in param.keys():
            if key not in common_algorithm_config:
                algorithm_config[key] = param[key]

        pickle.dump(problem_config, open(f"{config.BASE_PATH}/configs/problem_configs/{problem}.pickle", "wb"))
        pickle.dump(algorithm_config, open(f"{config.BASE_PATH}/configs/algorithm_configs/{problem}.pickle", "wb"))



def main():
    # problems = ["conceptual_marine_design"]
    find_preferred_id(problems)


if __name__ == '__main__':
    main()