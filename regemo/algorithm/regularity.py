from regemo.utils.algo_utils import fit_curve

import copy
import numpy as np


class Regularity():
    # class for every regularity
    def __init__(self,
                 dim,
                 lb,
                 ub,
                 fixed_vars,
                 fixed_vals,
                 non_fixed_vars,
                 non_fixed_dependent_vars,
                 non_fixed_independent_vars,
                 non_fixed_orphan_vars,
                 non_fixed_final_reg_coef_list,
                 non_fixed_degree_list,
                 problem_configs,
                 precision=2):

        self.dim = dim
        self.lb = lb
        self.ub = ub
        self.non_fixed_vars = non_fixed_vars
        self.fixed_vars = fixed_vars
        self.fixed_vals = fixed_vals
        self.non_fixed_dependent_vars = non_fixed_dependent_vars
        self.non_fixed_independent_vars = non_fixed_independent_vars
        self.non_fixed_orphan_vars = non_fixed_orphan_vars
        self.non_fixed_final_reg_coef_list = non_fixed_final_reg_coef_list
        self.non_fixed_degree_list = non_fixed_degree_list
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

    def apply(self, X):
        # enforce the final regularity to any vector of same size
        new_X = copy.deepcopy(X)

        # for fixed variables fix them to the found values
        new_X[:, self.fixed_vars] = self.fixed_vals

        # for non-fixed variables use the equation found in the process
        if self.non_fixed_dependent_vars and self.non_fixed_independent_vars:
            for i, dep_idx in enumerate(self.non_fixed_dependent_vars):
                new_X[:, dep_idx] = 0
                for j, list_degree in enumerate(self.non_fixed_degree_list):
                    new_X[:, dep_idx] += self.non_fixed_final_reg_coef_list[i][j] * \
                                         np.sum(
                                             np.power(new_X[:, self.non_fixed_independent_vars], np.array(list_degree)),
                                             axis=1)

                new_X[:, dep_idx] += self.non_fixed_final_reg_coef_list[i][-1]

        return new_X

    def fixed_regularity_repair(self, X_apply):
        # define the regularity repair procedure for fixed clusters of variables
        X = copy.deepcopy(X_apply)
        if len(X.shape) == 1:
            X = np.array([X])

        new_X = copy.deepcopy(X)
        new_X[:, self.fixed_vars] = self.fixed_vals

        return new_X

    def calc_process_complexity(self):
        # calculate the complexity of a particular regularity configuration
        # number of fixed, independent, dependent and orphan variables
        num_fixed = len(self.fixed_vars)
        num_independent = len(self.non_fixed_independent_vars)
        num_dependent = len(self.non_fixed_dependent_vars)
        num_orphan = len(self.non_fixed_orphan_vars)

        sum_degree = 0
        for i, dep_idx in enumerate(self.non_fixed_dependent_vars):
            # max_degree = 0
            for j, list_degree in enumerate(self.non_fixed_degree_list):
                if self.non_fixed_final_reg_coef_list[i][j] != 0:
                    cur_degree = 0
                    for idx, degree in enumerate(list_degree):
                        if degree != 0:
                            cur_degree += degree
                    # max_degree = max(max_degree, cur_degree)
                    sum_degree += cur_degree

            # sum_degree += max_degree

        # weight of the variables
        fixed_weight = 0.5
        # independent_weight = 3 * self.dim
        dependent_weight = 3 * num_independent
        orphan_weight = ((self.dim ** 4) + (5 * (self.dim ** 3)) + (6 * (self.dim ** 2)))/2    # here we are considering 3 to be the max_degree

        # compute the complexity
        complexity = (fixed_weight * num_fixed) + sum_degree + (orphan_weight * num_orphan) + self.dim

        return complexity

    def display(self, X_apply=None, lb=None, ub=None, save_file=None):
        self.print = self.mod_print()

        if save_file:
            f = open(save_file, "w")
            self.print = self.mod_print(f)

        # if there is some X, apply the regularity
        X = copy.deepcopy(X_apply)

        if X is not None:
            X = self.apply(X)

        # display final regularity
        self.print("\n=====================================")
        self.print("           Final Pattern             ")
        self.print("=====================================\n")

        if self.fixed_vars:
            self.print("Pattern for fixed variables")

            for var, val in zip(self.fixed_vars, self.fixed_vals):
                self.print(f"x[{var + 1}]: {val}")
                self.print()
        else:
            self.print("There is no fixed variables in the problem")

        if self.non_fixed_vars:
            self.print("Pattern for non-fixed variables")
            self.print(f"Non-fixed variables: {[(i + 1) for i in self.non_fixed_vars]}")
            self.print(f"Non-fixed independent variables: {[(i + 1) for i in self.non_fixed_independent_vars]}")
            self.print(f"Non-fixed dependent variables: {[(i + 1) for i in self.non_fixed_dependent_vars]}")
            for i, dep_idx in enumerate(self.non_fixed_dependent_vars):
                self.print(f"x[{dep_idx + 1}] = ", end="")
                for j, list_degree in enumerate(self.non_fixed_degree_list):
                    if self.non_fixed_final_reg_coef_list[i][j] != 0:
                        self.print(f"({self.non_fixed_final_reg_coef_list[i][j]} * ", end="")
                        for idx, degree in enumerate(list_degree):
                            if degree != 0:
                                self.print(f"x[{self.non_fixed_independent_vars[idx] + 1}]^{degree}", end="")

                        self.print(") + ", end="")
                self.print(f"({self.non_fixed_final_reg_coef_list[i][-1]})")

            self.print(f"Orphan non-fixed variables: {[(i + 1) for i in self.non_fixed_orphan_vars]}")

            self.print(f"\nRanges for the non-fixed variables")
            if self.non_fixed_independent_vars:
                self.print(f"Non-fixed independent variables")
                for var in self.non_fixed_independent_vars:
                    self.print(f"x[{var + 1}]: [{self.lb[var]}, {self.ub[var]}]")

            if self.non_fixed_orphan_vars:
                self.print(f"\nOrphan non-fixed variables")
                for var in self.non_fixed_orphan_vars:
                    self.print(f"x[{var + 1}]: [{self.lb[var]}, {self.ub[var]}]")

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
                    total_fronts=-1):

        self.print = self.mod_print()

        if save_file:
            f = open(save_file, "w")
            self.print = self.mod_print(f)

        # if there is some X, apply the regularity
        X = copy.deepcopy(X_apply)

        if X is not None:
            X = self.apply(X)

        self.print("\\begin{code}{\\textwidth}{1.2in}{codebackground}")

        if total_fronts > 1:
            self.print(f"Regular Front {front_num + 1}")

        if self.fixed_vars:
            const_list = self.fixed_vars

            if const_list:
                self.print("\\codeline{\\codedef{Fixed Variables}:}")
                for i, idx in enumerate(const_list):
                    self.print("\\codeline{\\codetab $x_{" + str(idx + 1) + "} = " + str(self.fixed_vals[i]) + "$}")

        else:
            self.print("\\codeline{\\codedef{Fixed Variables}: None}")

        if self.non_fixed_vars:
            if self.non_fixed_orphan_vars:
                self.print("\\codeline{\\codedef{Orphan Variables}:}")
                for idx in self.non_fixed_orphan_vars:
                    self.print("\\codeline{\\codetab $x_{" + str(idx + 1) + "} \\in [" + str(self.lb[idx]) + ", "
                                                                                                             "" + str(
                        self.ub[
                            idx]) + "]$}")
                # self.print("x_{" + str(self.non_fixed_orphan_vars[-1] + 1) + "}$")
            else:
                self.print("\\codeline{\\codedef{Orphan Variables}: None}")

            if self.non_fixed_independent_vars:
                self.print("\\codeline{\\codedef{Non-fixed Independent Variables}:}")
                for idx in self.non_fixed_independent_vars:
                    self.print("\\codeline{\\codetab $x_{" + str(idx + 1) + "} \\in [" + str(self.lb[idx]) + ", "
                               + str(self.ub[
                                         idx])
                               + "]$}")
            else:
                self.print("\\codeline{\\codedef{Non-fixed Independent Variables}: None}")

            if self.non_fixed_dependent_vars:
                self.print("\\codeline{\\codedef{Non-fixed Dependent Variables}:}")
                for i, dep_idx in enumerate(self.non_fixed_dependent_vars):
                    self.print("\\codeline{\\codetab $x_{" + str(dep_idx + 1) + "} = ", end="")
                    for idx, indep_idx in enumerate(self.non_fixed_independent_vars):
                        if self.non_fixed_final_reg_coef_list[i][idx] != 0:
                            self.print(
                                "(" + str({self.non_fixed_final_reg_coef_list[i][idx]}) + "x_{" + str(
                                    indep_idx + 1) + "}) + ", end="")
                    self.print(str(self.non_fixed_final_reg_coef_list[i][-1]) + "$}")
                # for idx in self.non_fixed_dependent_vars[:-1]:
                #     self.print("x_{" + str(idx + 1) + "}, \ ", end="")
                # self.print("x_{" + str(self.non_fixed_dependent_vars[-1] + 1) + "}$")
            else:
                self.print("\\codeline{\\codedef{Non-fixed Dependent Variables}: None}")

        self.print("\\end{code}")

        # if self.non_fixed_independent_vars:
        #     for var in self.non_fixed_independent_vars + self.non_fixed_orphan_vars:
        #         self.print("$x_{" + str(var + 1) + "} \\in [" + str(self.lb[var]) + ", " + str(self.ub[var]) + "]$")
        #
        #     self.print()
        #
        #     # if self.non_fixed_orphan_vars:
        #     #     for var in self.non_fixed_orphan_vars:
        #     #         self.print("x_{" + str(var+1) + "} \\in [" + str(self.lb[var]) + ", " + str(self.ub[var]) + "]")
        # self.print("\\end{lstlisting}\n")

        # if long_version:
        #     self.print("\\end{document}")

        if save_file:
            f.close()
            self.print = self.mod_print()

    def display_tex_long(self,
                         X_apply=None,
                         lb=None,
                         ub=None,
                         save_file=None,
                         front_num=-1,
                         total_fronts=-1):

        self.print = self.mod_print()

        if save_file:
            f = open(save_file, "w")
            self.print = self.mod_print(f)

        # if there is some X, apply the regularity
        X = copy.deepcopy(X_apply)

        if X is not None:
            X = self.apply(X)

        self.print("\documentclass[lettersize,journal]{IEEEtran}\n\\usepackage{fancyvrb}\n\\usepackage{"
                   "color}\n\\usepackage{xcolor}\n\\usepackage[most]{tcolorbox}\n\\def\codefont{\\fontfamily{"
                   "lmtt}\selectfont}\n\\newcommand{\\textcode}[1]{{\\normalfont\\codefont #1}}\n\\newcommand{\eqcode}["
                   "1]{"
                   "\textrm{\codefont #1}}\n\\newcommand{\prog}[1]{{\textcode{#1}}\\xspace}\n\\newcounter{"
                   "codelinecounter}[section]\n\\definecolor{codebackground}{rgb}{1.0,1.0,1.0}\n\\definecolor{"
                   "codebackgroundprogress}{rgb}{1.0,0.99,0.98}\n\\definecolor{codeframe}{rgb}{0.8,0.8,"
                   "0.8}\n\\definecolor{codegreen}{rgb}{0,0.4,0}\n\\definecolor{codeblue}{rgb}{0.25,0.25,"
                   "0.75}\n\\definecolor{codegray}{rgb}{0.5,0.5,0.5}\n\\def\codefontsize{\\fontsize{8.5}{"
                   "9}\selectfont}  % Ideally should be \footnotesize. Set value {8}{9}\n\\newcommand{\codeline}[1]{{"
                   "#1\par}}\n\\newcommand{\codelinenum}{\stepcounter{codelinecounter} {\color{codegray} \ifnum\value{"
                   "codelinecounter}<10 0\fi\arabic{codelinecounter}}\ }\n\\newcommand{\\numberedcodeline}[1]{{"
                   "\codelinenum \codeline{#1}}}\n\\newcommand{\codecomment}[1]{{\color{codegray} \# "
                   "#1}}\n\\newcommand{\codefixed}[1]{{\color{codeblue} #1}}\n\\newcommand{\codedef}[1]{{\color{"
                   "codegreen} #1}}\n\\newcommand{\codereturn}[1]{{\color{codegreen} #1}}\n\\newcommand{\codetab}{"
                   "~~}\n\\newcommand{\codeskip}{\smallskip}\n\\newenvironment{code}[3]{  % Params: width, height, "
                   "color\n\\setcounter{codelinecounter}{0}\n\\begin{tcolorbox}[\nwidth=#1,height=#2,\nvalign=center,"
                   "left=0pt,right=0pt,top=0pt,bottom=0pt,\ncolback=#3,colframe=codeframe,boxrule=0.5pt,"
                   "arc=0pt]\n\\codefont\n\\codefontsize\n}{\n\\end{tcolorbox}\n}\n\\begin{document}")

        self.print("\\begin{code}{\\textwidth}{1.2in}{codebackground}")

        if total_fronts > 1:
            self.print(f"Regular Front {front_num + 1}")

        if self.fixed_vars:
            const_list = self.fixed_vars

            if const_list:
                self.print("\\codeline{\\codedef{Fixed Variables}:}")
                for i, idx in enumerate(const_list):
                    if len(const_list) == 1:
                        self.print("\\codeline{\\codetab $", end="")
                        self.print("x_{" + str(idx + 1) + "} = " + str(self.fixed_vals[i]) + "$}")

                    elif i % 4 == 0:
                        self.print("\\codeline{\\codetab $", end="")
                        if i == len(const_list) - 1:
                            self.print("x_{" + str(idx + 1) + "} = " + str(self.fixed_vals[i]) + "$}")
                        else:
                            self.print("x_{" + str(idx + 1) + "} = " + str(self.fixed_vals[i]) + ", ", end="")
                    else:
                        if i == len(const_list) - 1:
                            self.print("x_{" + str(idx + 1) + "} = " + str(self.fixed_vals[i]) + "$}")
                        elif (i + 1) % 4 == 0:
                            self.print("x_{" + str(idx + 1) + "} = " + str(self.fixed_vals[i]) + ",$}")
                        else:
                            self.print("x_{" + str(idx + 1) + "} = " + str(self.fixed_vals[i]) + ", ", end="")
        else:
            self.print("\\codeline{\\codedef{Fixed Variables}: None}")

        if self.non_fixed_vars:
            if self.non_fixed_orphan_vars:
                self.print("\\codeline{\\codedef{Orphan Variables}:}")
                for idx in self.non_fixed_orphan_vars:
                    self.print("\\codeline{\\codetab $x_{" + str(idx + 1) + "} \\in [" + str(self.lb[idx]) + ", "
                                                                                                             "" + str(
                        self.ub[
                            idx]) + "]$}")
                # self.print("x_{" + str(self.non_fixed_orphan_vars[-1] + 1) + "}$")
            else:
                self.print("\\codeline{\\codedef{Orphan Variables}: None}")

            if self.non_fixed_independent_vars:
                self.print("\\codeline{\\codedef{Non-fixed Independent Variables}:}")
                for idx in self.non_fixed_independent_vars:
                    self.print("\\codeline{\\codetab $x_{" + str(idx + 1) + "} \\in [" + str(self.lb[idx]) + ", "
                               + str(self.ub[
                                         idx])
                               + "]$}")
            else:
                self.print("\\codeline{\\codedef{Non-fixed Independent Variables}: None}")

            if self.non_fixed_dependent_vars:
                self.print("\\codeline{\\codedef{Non-fixed Dependent Variables}:}")

                for i, dep_idx in enumerate(self.non_fixed_dependent_vars):
                    self.print("\\codeline{\\codetab $x_{" + str(dep_idx + 1) + "} = ", end="")
                    for j, list_degree in enumerate(self.non_fixed_degree_list):
                        if self.non_fixed_final_reg_coef_list[i][j] != 0:
                            if j > 0:
                                self.print(" + " if self.non_fixed_final_reg_coef_list[i][j] > 0 else
                                           " - ", end="")
                            else:
                                self.print("" if self.non_fixed_final_reg_coef_list[i][j] > 0 else
                                           " - ", end="")
                            self.print(abs(self.non_fixed_final_reg_coef_list[i][j]), end="")
                            for idx, degree in enumerate(list_degree):
                                if degree != 0:
                                    degree_str = str(degree) if degree > 1 else ""
                                    self.print(
                                        "x_{" + str(self.non_fixed_independent_vars[idx] + 1) + "}^{" + degree_str +
                                        "}", end="")

                    self.print(" + " if self.non_fixed_final_reg_coef_list[i][-1] > 0 else " - ", end="")
                    self.print(str(abs(self.non_fixed_final_reg_coef_list[i][-1])) + "$}")
                # for idx in self.non_fixed_dependent_vars[:-1]:
                #     self.print("x_{" + str(idx + 1) + "}, \ ", end="")
                # self.print("x_{" + str(self.non_fixed_dependent_vars[-1] + 1) + "}$")
            else:
                self.print("\\codeline{\\codedef{Non-fixed Dependent Variables}: None}")

        self.print("\\end{code}\n\\end{document}")

        # if self.non_fixed_independent_vars:
        #     for var in self.non_fixed_independent_vars + self.non_fixed_orphan_vars:
        #         self.print("$x_{" + str(var + 1) + "} \\in [" + str(self.lb[var]) + ", " + str(self.ub[var]) + "]$")
        #
        #     self.print()
        #
        #     # if self.non_fixed_orphan_vars:
        #     #     for var in self.non_fixed_orphan_vars:
        #     #         self.print("x_{" + str(var+1) + "} \\in [" + str(self.lb[var]) + ", " + str(self.ub[var]) + "]")
        # self.print("\\end{lstlisting}\n")

        # if long_version:
        #     self.print("\\end{document}")

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
