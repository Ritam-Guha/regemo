from regemo.utils.algo_utils import fit_curve

import copy
import numpy as np


class Regularity():
    # class for every regularity
    def __init__(self,
                 dim,
                 lb,
                 ub,
                 non_rand_cluster,
                 non_rand_vars,
                 non_rand_vals,
                 rand_vars,
                 rand_dependent_vars,
                 rand_independent_vars,
                 rand_orphan_vars,
                 rand_final_reg_coef_list,
                 problem_configs,
                 degree=1,
                 precision=2):

        self.dim = dim
        self.lb = lb
        self.ub = ub
        self.non_rand_cluster = non_rand_cluster
        self.rand_vars = rand_vars
        self.non_rand_vars = non_rand_vars
        self.non_rand_vals = non_rand_vals
        self.rand_dependent_vars = rand_dependent_vars
        self.rand_independent_vars = rand_independent_vars
        self.rand_orphan_vars = rand_orphan_vars
        self.rand_final_reg_coef_list = rand_final_reg_coef_list
        self.degree = degree
        self.precision = precision
        self.complexity = 0
        self.problem_configs = problem_configs
        self.print = self.mod_print()

    def generate_points(self,
                        n_points=1000):
        dim = len(self.lb)
        random_X = np.random.uniform(self.lb, self.ub, (n_points, dim))
        regular_X = self.apply(random_X, self.lb, self.ub)
        return regular_X

    def check_bound(self, X):
        mask = self.lb <= X <= self.ub
        mask = np.prod(mask, axis=1)
        return mask

    def apply(self, X, lb, ub):
        # function to apply the regularity to the population members in X

        # for non-random variables, fix them to the repaired mean values
        new_X = self.non_rand_regularity_repair(X, lb, ub, self.non_rand_cluster, self.degree, self.precision)

        # for random variables, use the equation found in the process
        if self.rand_dependent_vars and self.rand_independent_vars:
            for i, dep_idx in enumerate(self.rand_dependent_vars):
                # initialize every dependent variable to be zeros for the population members
                new_X[:, dep_idx] = 0

                for j, indep_idx in enumerate(self.rand_independent_vars):
                    # multiply the independent variable values with the corresponding variables
                    # to get the approximation for the dependent variable
                    new_X[:, dep_idx] += self.rand_final_reg_coef_list[i][j] * new_X[:, indep_idx]

                # finally, add the offset
                new_X[:, dep_idx] += self.rand_final_reg_coef_list[i][-1]

        # new_X = new_X[self.check_bound(new_X)]

        return new_X

    def non_rand_regularity_repair(self, X_apply, lb, ub, regularity_clusters, degree, precision):
        # define the regularity repair procedure for non-random clusters of variables
        X = copy.deepcopy(X_apply)
        if len(X.shape) == 1:
            # linear regression needs a 2D array
            X = np.array([X])

        new_X = copy.deepcopy(X)
        # take the mean of the population across the variables
        # mean_X = np.mean(X, axis=0)
        #
        # # repair the mean values based on the clusters and clip+round them
        # normalized_mean_X = self._normalize(mean_X, lb, ub)
        # normalized_regular_mean_X = fit_curve(normalized_mean_X, regularity_clusters, degree)
        # regular_mean_X = self._denormalize(normalized_regular_mean_X, lb, ub)
        # regular_mean_X = np.round(np.clip(regular_mean_X, lb, ub), precision)
        #
        # # for every cluster of non-random variables, fix all the population members
        # # to corresponding repaired mean values
        # for cluster in regularity_clusters:
        #     new_X[:, cluster] = regular_mean_X[cluster]
        #
        # if new_X.shape[0] == 1:
        #     # converting a single array back to a 1D array
        #     new_X = new_X[0, :]
        new_X[:, self.non_rand_vars] = self.non_rand_vals

        return new_X

    def calc_process_complexity(self):
        # calcuate the complexity of a particular regularity configuration

        # number of fixed, independent, dependent and complete random variables
        num_fixed = len(self.non_rand_vars)
        num_independent = len(self.rand_independent_vars)
        num_dependent = len(self.rand_dependent_vars)
        num_complete_random = len(self.rand_orphan_vars)

        # weight of the variables
        fixed_weight = 0.5
        independent_weight = 6 * self.dim - 11
        dependent_weight = 3 * num_independent
        complete_random_weight = independent_weight * (self.dim - 2) + 4

        # compute the complexity
        complexity = (fixed_weight * num_fixed) + (independent_weight * num_independent) + (dependent_weight *
                                                                                            num_dependent) + (
                             complete_random_weight * num_complete_random)

        return complexity

    def display(self, X_apply=None, lb=None, ub=None, save_file=None):
        def increment_cluster_indices(clusters):
            current_clusters = copy.deepcopy(clusters)
            for k, cluster in enumerate(current_clusters):
                if cluster:
                    current_clusters[k] = [(i + 1) for i in cluster]

            return current_clusters

        self.print = self.mod_print()

        if save_file:
            f = open(save_file, "w")
            self.print = self.mod_print(f)

        # if there is some X, apply the regularity
        X = copy.deepcopy(X_apply)

        if X is not None:
            X = self.apply(X, lb, ub)

        # display final regularity
        self.print("\n=====================================")
        self.print("           Final Pattern             ")
        self.print("=====================================\n")

        if self.non_rand_cluster:
            self.print("Pattern for non-random variables")
            self.print(f"Non-random variables: {increment_cluster_indices(self.non_rand_cluster)}")

            for i, cluster in enumerate(self.non_rand_cluster):
                self.print(f"Cluster {i + 1}: {increment_cluster_indices([cluster])}")
                for j in cluster:
                    if X is not None:
                        # when there's some X, calculate the regularity mean and insert that
                        self.print(f"X[{j + 1}]: {X[0, j]}")
                    else:
                        # display general regularity
                        self.print(f"X[{j + 1}]: mean(X[:, {j}])")
                self.print()
        else:
            self.print("There is no Non-Random variables in the problem")

        if self.rand_vars:
            self.print("Pattern for random variables")
            self.print(f"Random variables: {[(i + 1) for i in self.rand_vars]}")
            self.print(f"Random independent variables: {[(i + 1) for i in self.rand_independent_vars]}")
            self.print(f"Random dependent variables: {[(i + 1) for i in self.rand_dependent_vars]}")
            for i, dep_idx in enumerate(self.rand_dependent_vars):
                self.print(f"X[{dep_idx + 1}] = ", end="")
                for idx, indep_idx in enumerate(self.rand_independent_vars):
                    if self.rand_final_reg_coef_list[i][idx] != 0:
                        self.print(f"({self.rand_final_reg_coef_list[i][idx]} * X[{indep_idx + 1}]) + ", end="")
                self.print(f"({self.rand_final_reg_coef_list[i][-1]})")

            self.print(f"Orphan random variables: {[(i + 1) for i in self.rand_orphan_vars]}")

            self.print(f"\nRanges for the random variables")
            if self.rand_independent_vars:
                self.print(f"Random independent variables")
                for var in self.rand_independent_vars:
                    self.print(f"X[{var + 1}]: [{self.lb[var]}, {self.ub[var]}]")

            if self.rand_orphan_vars:
                self.print(f"\nOrphan random variables")
                for var in self.rand_orphan_vars:
                    self.print(f"X[{var + 1}]: [{self.lb[var]}, {self.ub[var]}]")

        self.print("\n======================================\n")
        self.complexity = self.calc_process_complexity()
        self.print(f"Final complexity of the regularity: {self.complexity}")
        self.print(f"Original lb: {self.problem_configs['lb']}")
        self.print(f"Original ub: {self.problem_configs['ub']}")
        self.print("\n======================================\n")

        if save_file:
            f.close()
            self.print = self.mod_print()

    def display_tex(self,
                    X_apply=None,
                    lb=None,
                    ub=None,
                    save_file=None,
                    front_num=-1,
                    total_fronts=-1,
                    long_version=True):
        def increment_cluster_indices(clusters):
            current_clusters = copy.deepcopy(clusters)
            for k, cluster in enumerate(current_clusters):
                if cluster:
                    current_clusters[k] = [(i + 1) for i in cluster]

            return current_clusters

        self.print = self.mod_print()

        if save_file:
            f = open(save_file, "w")
            self.print = self.mod_print(f)

        # if there is some X, apply the regularity
        X = copy.deepcopy(X_apply)

        if X is not None:
            X = self.apply(X, lb, ub)

        # display final regularity
        if long_version:
            self.print("\\documentclass{article}\n"
                       "\\usepackage[T1]{fontenc}\n"
                       "\\usepackage{algorithm}\n"
                       "\\usepackage[]{algpseudocode}\n"
                       "\\usepackage{pseudo}\n"
                       "\\usepackage{listings}\n")

        self.print("\\lstset{escapeinside={(*@}{@*)}}\n"
                   "\\lstdefinestyle{mystyle}{\n"
                   "basicstyle=\\ttfamily\\footnotesize,\n"
                   "}\n"
                   "\lstset{style=mystyle}\n")

        if long_version:
            self.print("\\begin{document}\n")

        self.print("\\begin{lstlisting}[mathescape, language=python]")

        if total_fronts > 1:
            self.print(f"Regular Front {front_num + 1}")

        if self.non_rand_cluster:
            const_list_clusters = increment_cluster_indices(self.non_rand_cluster)
            const_list = sum(const_list_clusters, [])

            if const_list:
                self.print(f"constant variables: $", end="")
                for idx in const_list[:-1]:
                    self.print("x_{" + str(idx) + "}, \ ", end="")
                self.print("x_{" + str(const_list[-1]) + "}$")

            for i, cluster in enumerate(self.non_rand_cluster):
                const_list = increment_cluster_indices([cluster])[0]
                if const_list:
                    if len(self.non_rand_cluster) > 1:
                        self.print(f"cluster {i + 1}: $", end="")
                    else:
                        self.print(f"$", end="")
                    for idx in const_list[:-1]:
                        self.print("x_{" + str(idx) + "}, \ ", end="")
                    self.print("x_{" + str(const_list[-1]) + "}$")

                for j in cluster:
                    if X is not None:
                        # when there's some X, calculate the regularity mean and insert that
                        self.print("$x_{" + str(j + 1) + "}: " + str(X[0, j]) + "$")
                    else:
                        # display general regularity
                        self.print("$x_{" + str(j + 1) + "}: mean(X[:, " + str(j) + "])")
                self.print()

            self.print()

        else:
            self.print("There is no Non-Random variables in the problem")

        if self.rand_vars:
            if self.rand_independent_vars:
                self.print("random independent variables: $", end="")
                for idx in self.rand_independent_vars[:-1]:
                    self.print("x_{" + str(idx + 1) + "}, \ ", end="")
                self.print("x_{" + str(self.rand_independent_vars[-1] + 1) + "}$")

            if self.rand_dependent_vars:
                self.print("random dependent variables: $", end="")
                for idx in self.rand_dependent_vars[:-1]:
                    self.print("x_{" + str(idx + 1) + "}, \ ", end="")
                self.print("x_{" + str(self.rand_dependent_vars[-1] + 1) + "}$")

            for i, dep_idx in enumerate(self.rand_dependent_vars):
                self.print("$x_{" + str(dep_idx + 1) + "} = ", end="")
                for idx, indep_idx in enumerate(self.rand_independent_vars):
                    if self.rand_final_reg_coef_list[i][idx] != 0:
                        self.print(
                            f"({self.rand_final_reg_coef_list[i][idx]} \\times " + "x_{" + str(indep_idx + 1) + "}) + ",
                            end="")
                self.print(f"{self.rand_final_reg_coef_list[i][-1]}$")

            if self.rand_orphan_vars:
                self.print("complete random variables: $", end="")
                for idx in self.rand_orphan_vars[:-1]:
                    self.print("x_{" + str(idx + 1) + "}, \ ", end="")
                self.print("x_{" + str(self.rand_orphan_vars[-1] + 1) + "}$")

            # if self.rand_independent_vars:
            for var in self.rand_independent_vars + self.rand_orphan_vars:
                self.print("$x_{" + str(var + 1) + "} \\in [" + str(self.lb[var]) + ", " + str(self.ub[var]) + "]$")

            self.print()

            # if self.rand_orphan_vars:
            #     for var in self.rand_orphan_vars:
            #         self.print("x_{" + str(var+1) + "} \\in [" + str(self.lb[var]) + ", " + str(self.ub[var]) + "]")
        self.print("\\end{lstlisting}\n")

        if long_version:
            self.print("\\end{document}")

        if save_file:
            f.close()
            self.print = self.mod_print()

    @staticmethod
    def _normalize(x, lb, ub):
        # function to normalize x between 0 and 1
        new_x = copy.deepcopy(x)

        if len(new_x.shape) == 1:
            new_x = np.array([new_x])

        for i in range(new_x.shape[1]):
            new_x[:, i] = (new_x[:, i] - lb[i]) / (ub[i] - lb[i])

        if new_x.shape[0] == 1:
            # converting a single array back to a 1D array
            new_x = new_x[0, :]

        return new_x

    @staticmethod
    def _denormalize(x, lb, ub):
        # function to denormalize a value between 0 and 1
        # to a value between lb and ub
        new_x = copy.deepcopy(x)

        if len(new_x.shape) == 1:
            new_x = np.array([new_x])

        for i in range(new_x.shape[1]):
            new_x[:, i] = lb[i] + (new_x[:, i] * (ub[i] - lb[i]))

        if new_x.shape[0] == 1:
            # converting a single array back to a 1D array
            new_x = new_x[0, :]
        return new_x

    @staticmethod
    def mod_print(save_file=None):
        # modify the print operation to save in files
        if save_file:
            def new_print(*args, end="\n"):
                for arg in args:
                    print(arg, end=end, file=save_file),
                    print

        else:
            def new_print(*args, end="\n"):
                for arg in args:
                    print(arg, end=end),
                    print

        return new_print
