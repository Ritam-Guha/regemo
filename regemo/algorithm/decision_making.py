import regemo.config as config
from regemo.utils.path_utils import create_dir

import numpy as np
import pickle
import matplotlib.pyplot as plt
import shutil

def trade_off_point_estimation(pf,
                               w_loss=1,
                               w_gain=3,
                               only_max_trade_off=True,
                               thres=0.0,
                               **kwargs):
    # function to estimate the highest trade-off point from a pareto front
    pf = np.array(pf)
    if pf.shape[0] == 1:
        if only_max_trade_off:
            return 0
        else:
            return [0]
    num_solutions = pf.shape[0]
    trade_off_vals = np.zeros(num_solutions)
    sort_idx = np.argsort(pf[:, 0])
    pf = pf[sort_idx, :]
    pf_norm = (pf - np.min(pf, axis=0)) / (np.max(pf, axis=0) - (np.min(pf, axis=0)))
    R_right = np.zeros(num_solutions)
    R_left = np.zeros(num_solutions)

    weight_ratio = w_loss / w_gain
    set_excluded_points = set()

    for i in range(num_solutions - 1):
        denom_right = (pf_norm[i, 1] - pf_norm[i + 1, 1])
        if denom_right < thres:
            set_excluded_points.add(i + 1)
        denom_left = (pf_norm[num_solutions - i - 1, 0] - pf_norm[num_solutions - i - 2, 0])
        if denom_left < thres:
            set_excluded_points.add(num_solutions - i - 1)

    valid_points = list(set(np.arange(pf.shape[0])) - set_excluded_points)

    for i in range(num_solutions - 1):
        denom_right = (pf_norm[i, 1] - pf_norm[i + 1, 1])
        num_right = pf_norm[i + 1, 0] - pf_norm[i, 0]
        R_right[i] = (weight_ratio * num_right) / denom_right

        denom_left = (pf_norm[num_solutions - i - 1, 0] - pf_norm[num_solutions - i - 2, 0])
        num_left = pf_norm[num_solutions - i - 2, 1] - pf_norm[num_solutions - i - 1, 1]
        R_left[num_solutions - i - 1] = (1 / weight_ratio) * (num_left / denom_left)

    for i in range(num_solutions):
        if i in valid_points:
            if i == 0:
                trade_off_vals[i] = R_right[0]
            elif i == num_solutions - 1:
                trade_off_vals[i] = R_left[-1]
            else:
                trade_off_vals[i] = (R_right[i] + R_left[i]) / 2

    sorted_trade_off = np.argsort(-trade_off_vals)

    # for i in range(num_solutions):
    #     fig, ax = plt.subplots()
    #     ax.scatter(pf_norm[:, 0], pf_norm[:, 1], c="blue", s=40, edgecolor="black")
    #     ax.scatter(pf_norm[i, 0], pf_norm[i, 1], c="red", s=40, edgecolor="black")
    #     ax.set_title(f"left: {np.round(R_left[i], 5)}, right: {np.round(R_right[i], 5)}, R: "
    #                  f"{np.round(trade_off_vals[i], 5)}")
    #     ax.grid(alpha=0.3)
    #     fig.show()

    if only_max_trade_off:
        return sort_idx[sorted_trade_off[0]]
    else:
        return sort_idx[sorted_trade_off]

def perform_decision_making(pf,
                            threshold=2):
    pop_size, n_obj = pf.shape
    idx_list = np.arange(pop_size)
    sorted_idx = np.argsort(pf[:, 0])
    pf = pf[sorted_idx, :]
    idx_list = idx_list[sorted_idx]

    if np.min(pf[:, 1]) > threshold:
        return idx_list[-1]
    else:
        valid_points = pf[:, 1] < threshold
        pf = pf[valid_points, :]
        idx_list = idx_list[valid_points]
        highest_trade_off_pt = trade_off_point_estimation(pf=pf)
        return idx_list[highest_trade_off_pt]


def runner(problem_name="bnh"):
    upper_level_search_path = f"{config.BASE_PATH}/results/upper_level_search/{problem_name}"
    upper_level_pf = pickle.load(open(f"{upper_level_search_path}/upper_level_pareto_front.pickle", "rb"))
    upper_level_pf = upper_level_pf["F"]

    # plot the Pareto front
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(upper_level_pf[:, 0], upper_level_pf[:, 1], s=40, edgecolor="black", label="RegEM(a)O NDTP")
    ax.set_xlabel("Complexity", fontsize=14)
    ax.set_ylabel("$\Delta$HV (in %)", fontsize=14)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")
    fig.savefig(f"{upper_level_search_path}/upper_level_pareto_front.pdf", format="pdf", bbox_inches="tight")
    fig.show()

    # decision-making
    shutil.rmtree(f"{upper_level_search_path}/selected_comb")
    selected_idx = perform_decision_making(pf=upper_level_pf)
    # create_dir(f"{upper_level_search_path.replace(config.BASE_PATH, '')}/selected_comb")
    shutil.copytree(f"{upper_level_search_path}/param_comb_{selected_idx+1}",
                    f"{upper_level_search_path}/selected_comb")

    # plot the Pareto front
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(upper_level_pf[:, 0], upper_level_pf[:, 1], s=40, edgecolor="black", label="RegEM(a)O NDTP")
    ax.scatter(upper_level_pf[selected_idx, 0], upper_level_pf[selected_idx, 1], s=40, c="red", edgecolor="black",
               label="DM Selection")
    ax.set_xlabel("Complexity", fontsize=14)
    ax.set_ylabel("$\Delta$HV (in %)", fontsize=14)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")
    fig.savefig(f"{upper_level_search_path}/upper_level_pareto_front_with_selection.pdf", format="pdf",
                bbox_inches="tight")
    fig.show()


def main():
    runner(problem_name="bnh")


if __name__ == "__main__":
    main()