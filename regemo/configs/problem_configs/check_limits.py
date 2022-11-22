from regemo.problems.get_problem import problems
import pickle

for problem in problems:
    config = pickle.load(open(f"{problem}.pickle", "rb"))
    print(problem)
    print(config["lb"])
    print(config["ub"])