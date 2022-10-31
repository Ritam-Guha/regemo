import regemo.config as config
from regemo.problems.get_problem import problems

import yaml
import pickle


def convert(file_path):
    config_pickle = pickle.load(open(file_path, "rb"))
    file_store = file_path.replace("pickle", "yaml")
    yaml.dump(config_pickle, open(file_store, "w+"))


def main():
    for problem in problems:
        convert(f"{config.BASE_PATH}/configs/algorithm_configs/{problem}.pickle")
        convert(f"{config.BASE_PATH}/configs/problem_configs/{problem}.pickle")


if __name__ == "__main__":
    main()