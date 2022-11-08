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
                    f"power_law: X[{i+1}] * X[{j+1}] ^ {np.round(power_law.b[i, j], 2)} = {np.round(power_law.c[i, j], 2)}",
                    end=" ", file=f)
                print(f", r2 score: {np.round(power_law.score_matrix[i, j], 2)}", file=f)

        return power_law, power_rules_idx

    except ValueError:
        print(f"could not learn rules for {problem_name}")


def plot_curves():
    color_map = {0: "blue",
                 1: "black"}
    fig, ax = plt.subplots()
    pop = get_population("bnh")
    power_law, power_rules_index = get_rules(problem_name="bnh")
    design_rep = pop["X"]
    ax.scatter(design_rep[:, 0], design_rep[:, 1], color="red", alpha=0.5, label="original points")
    for i, j in power_rules_index[:1]:
        lb = 0
        if i == 0:
            ub = 3
        else:
            ub = 5
        gen_points = np.random.uniform(low=lb, high=ub, size=1000)

        if i == 0:
            points = np.column_stack((power_law.c[i, j]/(gen_points ** power_law.b[i, j]), gen_points))
        else:
            points = np.column_stack((gen_points, power_law.c[i, j] / (gen_points ** power_law.b[i, j])))

        ax.plot(points[:, 0], points[:, 1], color=color_map[j],
                 label=f"innov: X[{i+1}] * X[{j+1}] ^ {np.round(power_law.b[i, j], 2)} = {np.round(power_law.c[i, j], 2)}, score: {np.round(power_law.score_matrix[i, j], 2)}")

    gen_points = np.random.uniform(low=0.03, high=4.18, size=1000)
    points = np.column_stack((gen_points, 0.6 * gen_points + 0.49))
    ax.plot(points[:, 0], points[:, 1], color="green", label="regemo: X[2] = 0.6 * X[1] + 0.49")

    plt.legend(loc="lower right")
    ax.set_xlabel("X[1]")
    ax.set_ylabel("X[2]")

    plt.savefig("design_space_comparison.svg", dpi=400)
    plt.show()


def main():
    # problem_name = "four_bar_truss_design"
    # for problem_name in problems:
    #     get_rules(problem_name)
    plot_curves()

if __name__ == "__main__":
    main()



