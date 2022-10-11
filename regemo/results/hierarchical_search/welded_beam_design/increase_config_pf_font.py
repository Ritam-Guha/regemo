import regemo.config as config

import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})


def get_config_pf_plot():
    problem_name = "welded_beam_design"
    df = pd.read_excel(f"{config.BASE_PATH}/results/hierarchical_search/{problem_name}/configurations.xlsx")
    pf_comb = [83, 92, 245, 254, 730]

    df = df[df["ID"].isin(pf_comb)]
    df["type"] = ["preferred"] + ["normal"]*4

    # matplotlib plots
    custom_color_mapping = {"preferred": "green", "normal": "blue"}
    type_wise_partition = {}
    for key in list(custom_color_mapping.keys()):
        type_wise_partition[key] = df[df["type"] == key][["complexity", "hv_dif_%"]].values

    fig, ax = plt.subplots(figsize=(10, 8))
    for key in list(custom_color_mapping.keys()):
        ax.scatter(type_wise_partition[key][:, 0], type_wise_partition[key][:, 1], c=custom_color_mapping[key],
                   label=key, s=80)

    ax.set_xlabel("complexity", fontsize=20)
    ax.set_ylabel("hv difference (in %)", fontsize=20)
    ax.legend(loc="upper right", fontsize=18, title_fontsize=20)

    fig.show()
    fig.savefig(f"{config.BASE_PATH}/results/hierarchical_search/{problem_name}/config_pf_plot.jpg", dpi=1200)


def main():
    get_config_pf_plot()


if __name__ == "__main__":
    main()