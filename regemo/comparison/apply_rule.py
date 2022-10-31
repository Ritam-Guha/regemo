import regemo.config as config
from regemo.comparison.relation import PowerLaw

import pickle


def get_population(problem_name):
    return pickle.load(open(f"{config.BASE_PATH}/results/hierarchical_search/{problem_name}/initial_population_{problem_name}.pickle", "rb"))


def get_rules(problem_name):
    pop = get_population(problem_name)
    design_rep = pop["X"]
    power_law = PowerLaw(n_var=design_rep.shape[1])
    power_law.learn(training_data=design_rep)
    print(power_law.b)


def main():
    problem_name = "bnh"
    get_rules(problem_name)


if __name__ == "__main__":
    main()



