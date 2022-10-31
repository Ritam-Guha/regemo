import regemo.config as config
from regemo.comparison.relation import PowerLaw
from regemo.problems.get_problem import get_problem, problems

import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score


def get_population(problem_name):
    return pickle.load(open(f"{config.BASE_PATH}/results/hierarchical_search/{problem_name}/initial_population_{problem_name}.pickle", "rb"))


def get_rules(problem_name):
    pop = get_population(problem_name)
    design_rep = pop["X"]
    power_law = PowerLaw(n_var=design_rep.shape[1])

    try:
        power_law.learn(training_data=design_rep)
        power_rules_idx = []
        scores = []

        for i in range(design_rep.shape[1]):
            for j in range(design_rep.shape[1]):
                if i != j:
                    power_rules_idx.append((i, j))
                    scores.append(power_law.score_matrix[i, j])

        scores = np.array(scores)
        idx = np.argsort(-scores)
        scores = scores[idx]
        power_rules_idx = np.array(power_rules_idx)[idx]

        with open(f'{config.BASE_PATH}/comparison/rules/{problem_name}.txt', 'w') as f:
            for i, j in power_rules_idx:
                print(
                    f"power_law: X[{i}] * X[{j}] ^ {np.round(power_law.b[i, j], 2)} = {np.round(power_law.c[i, j], 2)}",
                    end=" ", file=f)
                print(f"r2 score: {np.round(power_law.score_matrix[i, j], 2)}", file=f)

    except ValueError:
        print(f"could not learn rules for {problem_name}")


def main():
    # problem_name = "four_bar_truss_design"
    for problem_name in problems:
        get_rules(problem_name)


if __name__ == "__main__":
    main()



