from regemo.problems.get_problem import problems
import pickle


def get_range(feature):
    list_vals = []

    for problem in problems:
        algo_config = pickle.load(open(f"{problem}.pickle", "rb"))
        list_vals.append(algo_config["NSGA_settings"][feature])

    print(f"min: {min(list_vals)}, max:{max(list_vals)}")


def main():
    feature = "n_eval"
    get_range(feature)


if __name__ == "__main__":
    main()