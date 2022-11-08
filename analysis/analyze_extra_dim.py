import regemo.config as config
from regemo.problems.get_problem import problems

import pickle
import os


def get_name_problems():
    for problem in problems:
        reg_enf_files = [file for file in os.listdir(f"{config.BASE_PATH}/results/{problem}") if "random_var" in file]

        for file in reg_enf_files:
            try:
                count_random, count_dim = pickle.load(open(f"{config.BASE_PATH}/results/{problem}/{file}", "rb"))
            except:
                print(f"EOF for {problem}")
                break

            if count_random > count_dim-1:
                print(problem)
                break


def main():
    get_name_problems()


if __name__ == '__main__':
    main()

