import regemo.config as config

import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 30})


def get_config_pf_plot():
    df = pd.read_excel(f"{config.BASE_PATH}/results/hierarchical_search/bnh/configurations.xlsx")
    pf_comb = [244, 253]

    df = df[df["ID"].isin(pf_comb)]
    df["type"] = ["normal", "preferred"]

    # matplotlib plots
    custom_color_mapping = {"preferred": "green", "knee": "red", "normal": "blue"}
    type_wise_partition = {}
    for key in list(custom_color_mapping.keys()):
        type_wise_partition[key] = df[df["type"] == key][["complexity", "hv_dif_%"]].values

    fig, ax = plt.subplots(figsize=(10, 9))
    for key in list(custom_color_mapping.keys()):
        ax.scatter(type_wise_partition[key][:, 0], type_wise_partition[key][:, 1], c=custom_color_mapping[key],
                   label=key, s=120)

    ax.set_xlabel("complexity", fontsize=30)
    ax.set_ylabel("hv difference (in %)", fontsize=30)
    ax.legend(loc="upper right", fontsize=30, title_fontsize=30)

    fig.show()
    fig.savefig(f"{config.BASE_PATH}/results/hierarchical_search/bnh/config_pf_plot.jpg", dpi=600)


def main():
    get_config_pf_plot()


if __name__ == "__main__":
    main()