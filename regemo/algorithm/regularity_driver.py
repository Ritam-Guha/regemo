import regemo.config as config
from regemo.algorithm.regularity_search import Regularity_Search
from regemo.problems.get_problem import problems as problem_set
from regemo.utils.path_utils import create_dir

from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.misc import find_duplicates
from pymoo.factory import get_performance_indicator

import copy
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import pickle
import os
from tabulate import tabulate
import sys
import argparse
import plotly.express as px

plt.rcParams.update({'font.size': 15})


class Regularity_search_driver:
    def __init__(self,
                 problem_args,
                 algorithm_args,
                 exec_list,
                 seed=1,
                 verbose=True):
        """
        :param problem_args: parameters of the problem
        :param algorithm_args: parameters of the algorithm (NSGA2/NSGA3)
        :param exec_list: list of values for the hyperparameters
        :param seed: seed assigned to the runs
        :param verbose: whether to print the details of the runs to the console
        """
        # store the passed arguments
        self.problem_args = problem_args
        self.algorithm_args = algorithm_args
        self.exec_list = exec_list
        self.seed = seed
        self.verbose = verbose

        # create proxy variables
        self.root_dir = f"results/hierarchical_search"
        self.problem_name = self.problem_args["problem_name"]
        self.param_comb = None
        self.pf_param_comb = None
        self.knee_point_ID = -10
        self.convexity_front = None
        self.preferred_ID = None

        # set the run seed
        np.random.seed(self.seed)

    def create_file_structure(self):
        # create the file structure for storing the results
        create_dir(self.root_dir, delete=False)
        problem_path = f"{self.root_dir}/{self.problem_args['problem_name']}"
        create_dir(problem_path, delete=True)

        # create a folder for every parameter combination
        for i in range(len(self.param_comb)):
            final_folder_path = f"{problem_path}/param_comb_{i + 1}"
            create_dir(final_folder_path)

    def run(self):
        # the main running part
        self.param_comb = self.find_param_comb()    # recursively find all the hyperparameter configs
        self.create_file_structure()                # create the file structure for storing results
        self.execute_regularity_search()            # execute the reg search wrt the hyperparameter configs
        self.plot_complexity_vs_efficiency()        # compute the trade-off between complexity and efficiency
        self.save_config_df()                       # store the final config results in files
        self.perform_trade_off_analysis()           # perform analysis on the trade-off of the final results

    def find_param_comb(self):
        # driver function to recursively find the list of parameters
        list_keys = list(self.exec_list.keys())
        for key in list_keys:
            if not isinstance(self.exec_list[key], list):
                self.exec_list[key] = [self.exec_list[key]]

        list_param_comb = []
        self._rec_list(self.exec_list, list_keys, 0, {}, list_param_comb)
        return list_param_comb

    def _rec_list(self, params, list_keys, cur_key_idx, cur_list, list_param_comb):
        # recursively finds all the parameter combinations
        if cur_key_idx == len(list_keys):
            list_param_comb.append(cur_list)
            return

        list_vals = params[list_keys[cur_key_idx]]

        for val in list_vals:
            cur_list[list_keys[cur_key_idx]] = val
            self._rec_list(params, list_keys, cur_key_idx + 1, cur_list.copy(), list_param_comb)
            del cur_list[list_keys[cur_key_idx]]

    def execute_regularity_search(self):
        table_header = ["Id", "coef_factor", "non_fixed_dependency_percent", "complexity", "hv_dif_%"]
        table_data = []

        print("=============================================")
        print(f"              {self.problem_name}           ")
        print("=============================================")
        # execute the process for every regularity combination

        for i, param in enumerate(self.param_comb):
            print(f"Config ID: {i}, Algorithm Config: {param}")
            cur_ps = Regularity_Search(problem_args=self.problem_args,
                                       non_fixed_regularity_coef_factor=param["non_fixed_regularity_coef_factor"],
                                       non_fixed_dependency_percent=param["non_fixed_dependency_percent"],
                                       n_rand_bins=param["n_rand_bins"],
                                       delta=param["delta"],
                                       seed=self.seed,
                                       NSGA_settings=self.algorithm_args["NSGA_settings"],
                                       result_storage=(
                                           f"{self.root_dir}/{self.problem_args['problem_name']}/param_comb_{i + 1}"),
                                       verbose=self.verbose)
            cur_ps.run()    # run the regularity search object

            # add the additional information about the param config
            param["ID"] = i + 1
            param["complexity"] = cur_ps.final_metrics["complexity"]
            param["hv_dif_%"] = cur_ps.final_metrics["hv_dif_%"]

            table_data.append([param["ID"], param["non_fixed_regularity_coef_factor"],
                               param["non_fixed_dependency_percent"],
                               param["complexity"], param["hv_dif_%"]])

            # save the param config
            self.save_param_config(param)
            print(f"Param Config {i} completed.\n")
            plt.close("all")

        print(f"\n{tabulate(table_data, headers=table_header)}\n")

    def save_param_config(self, param):
        # store the parameter obj to a pickle file
        final_folder_path = f"{self.root_dir}/{self.problem_name}/param_comb_{param['ID']}"
        with open(f"{config.BASE_PATH}/{final_folder_path}/param_comb.pkl", "wb") as pkl_handle:
            pickle.dump(param, pkl_handle)

        # store the parameter config in a text file for easier visualization
        text_file = open(f"{config.BASE_PATH}/{final_folder_path}/parameter_configuration.txt", "w")
        for key in param.keys():
            print(key + "= " + str(param[key]), file=text_file)

    def plot_complexity_vs_efficiency(self):
        # plot the complexity vs efficiency curve
        fig = plt.figure()
        for param in self.param_comb:
            plt.scatter(param["complexity"], param["hv_dif_%"], c="b")

        plt.xlabel("complexity")
        plt.ylabel("hv_dif_%")
        fig.savefig(f"{config.BASE_PATH}/{self.root_dir}/{self.problem_name}/param_comb_trade_off.pdf", format="pdf")

    @staticmethod
    def knee_point_estimation(pf, w_loss=1, w_gain=1):
        # function to estimate the highest trade-off point from a pareto front
        num_solutions = np.array(pf).shape[0]
        R = np.zeros(num_solutions)
        trade_off_vals = np.zeros(num_solutions)
        pf_norm = (pf - np.min(pf, axis=0)) / (np.max(pf, axis=0) - (np.min(pf, axis=0)) + 1e-6)

        for i in range(num_solutions - 1):
            R[i] = (w_loss * (pf_norm[i + 1, 0] - pf_norm[i, 0])) / (w_gain * (pf_norm[i, 1] - pf_norm[i + 1, 1]))

        for i in range(num_solutions):
            if i == 0:
                trade_off_vals[i] = R[0]
            elif i == num_solutions - 1:
                trade_off_vals[i] = R[-1]
            else:
                trade_off_vals[i] = np.mean(R[i - 1:i + 1])

        trade_off_point = np.argmax(trade_off_vals)

        return trade_off_point

    def find_preferred_point(self,
                             threshold_percentage=2):
        pf = np.zeros((len(self.pf_param_comb), 2))
        for i, comb in enumerate(self.pf_param_comb):
            pf[i, 0] = comb["complexity"]
            pf[i, 1] = comb["hv_dif_%"]

        sorted_idx = np.argsort(pf[:, 0])
        self.pf_param_comb = [self.pf_param_comb[i] for i in sorted_idx]
        pf = pf[sorted_idx, :]  # sort the pareto front in increasing order of first objective

        # check if the pareto front is convex
        self.convexity_front = self.is_convex_front(pf)

        # find the knee point
        knee_point_index = self.knee_point_estimation(pf)
        self.knee_point_ID = self.pf_param_comb[knee_point_index]["ID"]

        # get the pareto front under the threshold
        threshold_idx = [i for i in range(pf.shape[0]) if pf[i, 1] <= threshold_percentage]
        pf_threshold = pf[threshold_idx, :]
        self.pf_param_comb_threshold = [self.pf_param_comb[idx] for idx in threshold_idx]

        if len(threshold_idx) == 0:
            return self.pf_param_comb[-1]["ID"]
        elif pf[knee_point_index, 1] <= threshold_percentage:
            # the preferred point is the knee point, if it is within 2% error
            return self.knee_point_ID
        elif pf_threshold.shape[0] > 0:
            knee_point_index_threshold = self.knee_point_estimation(pf_threshold)
            return self.pf_param_comb_threshold[knee_point_index_threshold]["ID"]

    def perform_trade_off_analysis(self):
        pf_df = pd.read_excel(f"{config.BASE_PATH}/{self.root_dir}/{self.problem_name}/configurations.xlsx",
                              sheet_name="All Configs")
        self.param_comb = []
        for index, row in pf_df.iterrows():
            cur_dict = {}
            for col in pf_df.columns:
                cur_dict[col] = row[col]

            self.param_comb.append(cur_dict)

        F = np.zeros((pf_df.shape[0], 2))
        F[:, 0] = pf_df["complexity"].to_numpy()
        F[:, 1] = pf_df["hv_dif_%"].to_numpy()

        # perform non-dominated sorting
        nds = NonDominatedSorting()
        fronts = nds.do(F)
        F = F[fronts[0], :]
        self.pf_param_comb = [copy.deepcopy(self.param_comb[i]) for i in fronts[0]]

        # need to retain the unique variations
        # because if they are not unique, they lead division-by-zero in knee point calculation
        is_unique = np.where(np.logical_not(find_duplicates(F, epsilon=1e-24)))[0]
        F = F[is_unique, :]
        self.pf_param_comb = [self.pf_param_comb[i] for i in is_unique]
        thresholded_percentage = 1
        self.preferred_ID = self.find_preferred_point(threshold_percentage=thresholded_percentage)

        pf_text = open(f"{config.BASE_PATH}/{self.root_dir}/{self.problem_name}/config_PF.txt", "w")
        for param in self.pf_param_comb:
            if param["ID"] == self.knee_point_ID or param["ID"] == self.preferred_ID:
                if param["ID"] == self.knee_point_ID:
                    print("---------------------------------------------", file=pf_text)
                    print("                  Knee Point                 ", file=pf_text)
                    print("---------------------------------------------", file=pf_text)
                    param["type"] = "knee"
                    for key in param.keys():
                        print(key + "= " + str(param[key]), file=pf_text)
                    print("---------------------------------------------", file=pf_text)

                if param["ID"] == self.preferred_ID:
                    print("---------------------------------------------", file=pf_text)
                    print("                Preferred Point              ", file=pf_text)
                    print("---------------------------------------------", file=pf_text)
                    param["type"] = "preferred"
                    for key in param.keys():
                        print(key + "= " + str(param[key]), file=pf_text)
                    print("---------------------------------------------", file=pf_text)

            else:
                print("---------------------------------------------", file=pf_text)
                param["type"] = "normal"

                for key in param.keys():
                    print(key + "= " + str(param[key]), file=pf_text)
                print("---------------------------------------------", file=pf_text)

        # save the final pareto front plot
        x_title = "complexity"
        y_title = "hv_dif_%"
        hover_data = list(self.param_comb[0].keys())
        pf_df = self.convert_pf_regularity_to_pd(self.pf_param_comb)
        pickle.dump(self.preferred_ID, open(f"{config.BASE_PATH}/results/hierarchical_search/{self.problem_name}/"
                                            f"preferred_id.pickle", "wb"))

        # save the interactive pf config
        custom_color_mapping = {"preferred": "green", "knee": "red", "normal": "blue"}
        fig = px.scatter(pf_df, x=x_title, y=y_title, color="type", hover_data=hover_data, log_x=True, size_max=60,
                         color_discrete_map=custom_color_mapping)
        fig.update_traces(textposition='top center', marker_size=10)
        fig.update_layout(
            height=800,
            title_text='Final Pareto Front Configuration for ' + self.problem_name
        )
        fig.show()
        fig.write_html(f"{config.BASE_PATH}/{self.root_dir}/{self.problem_name}/config_pf.html")
        fig.write_image(f"{config.BASE_PATH}/{self.root_dir}/{self.problem_name}/config_pf.pdf", format="pdf")

        # matplotlib plots
        custom_color_mapping = {"preferred": "green", "knee": "red", "normal": "blue"}
        type_wise_partition = {}
        for key in list(custom_color_mapping.keys()):
            type_wise_partition[key] = pf_df[pf_df["type"] == key][["complexity", "hv_dif_%"]].values

        fig, ax = plt.subplots(figsize=(10, 8))
        for key in list(custom_color_mapping.keys()):
            ax.scatter(type_wise_partition[key][:, 0], type_wise_partition[key][:, 1], c=custom_color_mapping[key],
                       label=key, s=80)
        ax.set_xlabel("complexity")
        ax.set_ylabel("hv difference (in %)")
        ax.legend(loc="upper right")
        fig.show()
        fig.savefig(f"{config.BASE_PATH}/{self.root_dir}/{self.problem_name}/config_pf_plot.pdf", format="pdf")

    @staticmethod
    def convert_pf_regularity_to_pd(param_comb):
        # convert the final PF regularities to a pandas dataframe
        df_dict = {}
        # first create dictionary with keys from the execution lists having empty lists
        for key in param_comb[0].keys():
            df_dict[key] = []

        # populate the empty lists with the param values
        for param in param_comb:
            for key in df_dict.keys():
                df_dict[key].append(param[key])

        pf_df = pd.DataFrame.from_dict(df_dict)

        return pf_df

    def save_config_df(self):
        # convert the param combinations
        df = self.convert_pf_regularity_to_pd(self.param_comb)

        # save the final info to an excel file at the end
        list_cols = list(df.columns)
        list_cols.remove("ID")
        ordered_list_cols = ["ID"] + list_cols

        with pd.ExcelWriter(f"{config.BASE_PATH}/{self.root_dir}/{self.problem_name}/configurations.xlsx") as writer:
            df.to_excel(writer, sheet_name='All Configs', index=False, columns=ordered_list_cols)

    @staticmethod
    def is_convex_front(F):
        # function to check if the front is convex
        n_pop = F.shape[0]

        # 3 or less-point boundaries are always concave
        if n_pop < 4:
            return True

        sign = None

        for i in range(n_pop):
            dx1 = F[(i + 1) % n_pop, 0] - F[i, 0]
            dx2 = F[(i + 2) % n_pop, 0] - F[(i + 1) % n_pop, 0]
            dy1 = F[(i + 1) % n_pop, 1] - F[i, 1]
            dy2 = F[(i + 2) % n_pop, 1] - F[(i + 1) % n_pop, 1]

            prod = dx1 * dy2 - dx2 * dy1

            if i == 0:
                sign = prod > 0
            elif sign != (prod > 0):
                return False

        return True


def main():
    # collect arguments for the problem
    seed = config.seed
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem_name", default="bnh", help="Name of the problem")
    args = parser.parse_args()
    problem_name = args.problem_name
    if problem_name != "all":
        # if you want to run it on a specific problem
        problems = [problem_name]
    else:
        # if you want to run it on all the problems in regemo suite
        problems = problem_set

    # for the specified problems, run the regularity driver
    for problem_name in problems:
        algorithm_config_storage_dir = f"{config.BASE_PATH}/{config.algorithm_config_path}"
        problem_config_storage_dir = f"{config.BASE_PATH}/{config.problem_config_path}"
        use_existing_config = True
        algorithm_config, problem_config = {}, {}

        if use_existing_config:
            if not os.path.exists(f"{problem_config_storage_dir}/{problem_name}.pickle"):
                print("[Error!] Problem Configuration file not found...")
                sys.exit(1)
            if not os.path.exists(f"{algorithm_config_storage_dir}/{problem_name}.pickle"):
                print("[Error!] Algorithm Configuration file not found...")
                sys.exit(1)
            else:
                problem_config = pickle.load(open(f"{problem_config_storage_dir}/{problem_name}.pickle", "rb"))
                algorithm_config = pickle.load(open(f"{algorithm_config_storage_dir}/{problem_name}.pickle", "rb"))

        problem_config["problem_name"] = problem_name

        # mention the possible values for the hyperparameters
        exec_args = {"non_fixed_regularity_coef_factor": [0.1, 0.3, 0.5],
                     "non_fixed_dependency_percent": [0.1, 0.3, 0.5, 0.7],
                     "delta": [0.05, 0.1, 0.2],
                     "n_rand_bins": [3, 4, 5, 10]}

        # create the driver object
        driver = Regularity_search_driver(problem_args=problem_config,
                                          algorithm_args=algorithm_config,
                                          exec_list=exec_args,
                                          seed=seed,
                                          verbose=False)

        # run the driver object
        driver.run()


if __name__ == "__main__":
    main()

