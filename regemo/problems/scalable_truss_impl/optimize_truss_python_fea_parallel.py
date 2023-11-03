import argparse
import logging
import multiprocessing as mp
import os
import dill as pickle
import shutil
import sys
import time
import warnings
import datetime
import regemo.config as config

import h5py
import matplotlib.pyplot as plt
import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_sampling, get_crossover, get_mutation, get_termination
from pymoo.optimize import minimize
from pymoo.util.display import MultiObjectiveDisplay

from regemo.problems.scalable_truss_impl.truss.truss_problem_general import TrussProblemGeneral
from regemo.problems.scalable_truss_impl.utils.logutils import setup_logging
import regemo.problems.scalable_truss_impl.utils.record_data_legacy as rec
from regemo.problems.scalable_truss_impl.utils.general import get_knee

time_now = datetime.datetime.now()
results_parent_dir = 'output'
save_folder = os.path.join(results_parent_dir, 'truss_optimization_nsga2')


class OptimizationDisplay(MultiObjectiveDisplay):

    def _do(self, problem, evaluator, algorithm):
        super()._do(problem, evaluator, algorithm)

        f = algorithm.pop.get("F")
        f_min_weight = np.round(f[f[:, 0] == np.min(f[:, 0]), :], decimals=2).flatten()
        f_min_compliance = np.round(f[f[:, 1] == np.min(f[:, 1]), :], decimals=2).flatten()
        self.output.append("Min. weight solution", f_min_weight)
        self.output.append("Min. compliance solution", f_min_compliance)
        self.output.append("cv(min.)", np.min(algorithm.pop.get('CV')))
        self.output.append("cv(max.)", np.max(algorithm.pop.get('CV')))

        logging.info("===============================================================================================")
        logging.info("n_gen |  n_eval | Min. weight solution      | Min. compliance solution      |   cv(min.)   |   "
                     "cv(max.)  ")
        logging.info(f"{algorithm.n_gen}    | {algorithm.n_gen * algorithm.pop_size}      | "
                     f"{f_min_weight} | {f_min_compliance} |   {np.min(algorithm.pop.get('CV'))}   |   "
                     f"{np.max(algorithm.pop.get('CV'))}  ")
        logging.info("===============================================================================================")


def parse_args(args):
    """Defines and parses the command line arguments that can be supplied by the user.

    Args:
        args (list): Command line arguments supplied by the user.

    """
    # Command line args accepted by the program
    parser = argparse.ArgumentParser(description='Large Scale Truss Design Optimization')

    # Truss parameters
    parser.add_argument('--nshapevar', type=int, default=19, help='Number of shape variables')
    parser.add_argument('--symmetric', action='store_true', default=False, help='Enforce symmetricity of trusses')

    # Optimization parameters
    parser.add_argument('--nruns', type=int, default=1, help='Total number of runs.')
    parser.add_argument('--seed', type=int, default=184716924, help='Random seed')
    parser.add_argument('--ngen', type=int, default=200, help='Maximum number of generations')
    parser.add_argument('--popsize', type=int, default=100, help='Population size')

    # Innovization
    # parser.add_argument('--innovization', action='store_true', default=False, help='Apply custom innovization operator')
    # parser.add_argument('--momentum', type=float, default=0.3, help='Value of momentum coefficient')
    # parser.add_argument('--interactive', action='store_true', default=False,
    #                     help='Enable interactive mode. Might interfere with online innovization')
    # parser.add_argument('--user-input-freq', type=float, default=100, help='Frequency with which user input is taken.')
    # parser.add_argument('--probability-update-freq', type=float, default=20, help='Frequency with which probability of selection of user provided operators is updated.')

    # Innovization
    parser.add_argument('--repair-inequality', action='store_true', default=False,
                        help='Apply custom innovization operator based on inequalities')
    # parser.add_argument('--repair-inequality-interval', type=int, default=5,
    #                     help='Frequency with which repair_inequality is performed.')
    parser.add_argument('--repair-power', action='store_true', default=False,
                        help='Apply custom innovization operator based on power laws')
    # parser.add_argument('--repair-power-interval', type=int, default=5,
    #                     help='Frequency with which repair_inequality is performed.')
    parser.add_argument('--repair-interval', type=int, default=10,
                        help='Frequency with which repair_inequality is performed.')

    # Logging parameters
    parser.add_argument('--save', type=str, help='Experiment name')

    # Parallelization
    parser.add_argument('--ncores', type=int, default=mp.cpu_count()//4,
                        help='How many cores to use for population members to be evaluated in parallel')

    # Not yet operational
    parser.add_argument('--report-freq', type=float, default=10, help='Default logging frequency in generations')
    parser.add_argument('--crossover', default='real_sbx', help='Choose crossover operator')
    parser.add_argument('--mutation-eta', default=20, help='Define mutation parameter eta')
    parser.add_argument('--mutation-prob', default=0.005, help='Define mutation parameter eta')

    return parser.parse_args(args)


if __name__ == '__main__':
    t0 = time.time()

    cmd_args = parse_args(sys.argv[1:])

    repair_str = 'repair'
    if cmd_args.repair_power:
        repair_str += f'_power_freq{cmd_args.repair_interval}'
    elif cmd_args.repair_inequality:
        repair_str += f'_inequality_freq{cmd_args.repair_interval}'
    else:
        repair_str = 'baseline'
    # if cmd_args.save is not None:
    #     save_folder = os.path.join(results_parent_dir, cmd_args.save)
    # else:
    folder_str = f'nshape{cmd_args.nshapevar}'
    if cmd_args.symmetric:
        folder_str += '_symm'
    folder_str += f'_{cmd_args.popsize}pop_{cmd_args.ngen}gen_{repair_str}_{time_now.strftime("%Y%m%d_%H%M%S")}'

    random_seed_list = np.loadtxt(f'{config.BASE_PATH}/problems/scalable_truss_impl/random_seed_list')
    starting_run = 1  # If some runs have already completed beforehand, we can start from a later point
    for i in range(starting_run - 1, cmd_args.nruns):
        plt.close('all')
        if cmd_args.nruns == 1:
            seed = cmd_args.seed
        else:
            seed = int(random_seed_list[i])

        print(f"Run {i + 1}, seed {seed}")
        save_folder = os.path.join(results_parent_dir, folder_str, f"run{i + 1}_seed{seed}")
        if cmd_args.symmetric:
            symmetry = ('xz', 'yz')
        else:
            symmetry = ()
        truss_problem = TrussProblemGeneral(n_shape_var=cmd_args.nshapevar,
                                            n_cores=cmd_args.ncores,
                                            repair_inequality=cmd_args.repair_inequality,
                                            repair_power=cmd_args.repair_power,
                                            repair_interval=cmd_args.repair_interval,
                                            symmetry=symmetry
                                            )
        print(f"Full symmetric truss {truss_problem}")

        truss_optimizer = NSGA2(
            pop_size=cmd_args.popsize,
            sampling=get_sampling("real_random"),
            crossover=get_crossover("real_sbx", prob=0.9, eta=3),
            mutation=get_mutation("real_pm", eta=3),
            eliminate_duplicates=True,
            callback=rec.record_state,
            display=OptimizationDisplay()
        )
        termination = get_termination("n_gen", cmd_args.ngen)

        if os.path.exists(save_folder):
            shutil.move(save_folder, os.path.join(results_parent_dir, f'backup_{time.strftime("%Y%m%d-%H%M%S")}',
                                                  os.path.basename(save_folder)
                                                  ))
        os.makedirs(save_folder)

        rec.results_dir = save_folder

        with open(os.path.join(save_folder, 'optim_args'), 'w') as fptr:
            fptr.write(str(cmd_args))
        setup_logging(save_folder)

        logging.info(f"User-supplied arguments: {sys.argv[1:]}")
        logging.info(f"All arguments after parsing: {cmd_args}")
        logging.info(f"Population size = {truss_optimizer.pop_size}, Max. generations = {termination.n_max_gen}")
        logging.info(f"Vars = {truss_problem.n_var}, "
                     f"Objectives = {truss_problem.n_obj}, Constraints = {truss_problem.n_constr}")
        logging.info(f"Range of decision variables:\nX_L=\n{truss_problem.xl}\nX_U=\n{truss_problem.xu}\n")
        logging.info(f"Size variables = {truss_problem.n_size_var}")
        logging.info(f"Shape variables = {truss_problem.n_shape_var}")
        logging.info(f"Fixed nodes:\n{truss_problem.fixed_nodes}")
        logging.info(f"Load nodes:\n{truss_problem.load_nodes}")
        logging.info(f"Force:\n{truss_problem.force}")

        logging.info("Beginning optimization")
        res = minimize(truss_problem,
                       truss_optimizer,
                       termination,
                       seed=seed,
                       save_history=False,
                       verbose=True)

        logging.info("Optimization complete. Writing data")
        print(res.F)

        # Save results
        # For final PF
        np.savetxt(os.path.join(save_folder, 'f_max_gen'), res.F)
        np.savetxt(os.path.join(save_folder, 'x_max_gen'), res.X)
        if truss_problem.n_constr > 0:
            np.savetxt(os.path.join(save_folder, 'g_max_gen'), res.G)
            np.savetxt(os.path.join(save_folder, 'cv_max_gen'), res.CV)

        # For final pop
        np.savetxt(os.path.join(save_folder, 'f_pop_max_gen'), res.pop.get('F'))
        np.savetxt(os.path.join(save_folder, 'x_pop_max_gen'), res.pop.get('X'))
        if truss_problem.n_constr > 0:
            np.savetxt(os.path.join(save_folder, 'g_pop_max_gen'), res.pop.get('G'))
            np.savetxt(os.path.join(save_folder, 'cv_pop_max_gen'), res.pop.get('CV'))
        np.savetxt(os.path.join(save_folder, 'rank_pop_max_gen'), res.pop.get('rank'))

        # Additional data for final pop
        num_members = res.pop[0].data['stress'].shape[0]
        num_nodes = res.pop[0].data['coordinates'].shape[0]
        stress_final_pop = np.zeros([truss_optimizer.pop_size, num_members])
        strain_final_pop = np.zeros([truss_optimizer.pop_size, num_members])
        u_final_pop = np.zeros([truss_optimizer.pop_size, num_nodes * 6])
        x0_new_final_pop = np.zeros([truss_optimizer.pop_size, num_nodes, 3])
        for indx in range(truss_optimizer.pop_size):
            stress_final_pop[indx, :] = res.pop[indx].data['stress'].reshape(1, -1)
            strain_final_pop[indx, :] = res.pop[indx].data['strain'].reshape(1, -1)
            u_final_pop[indx, :] = res.pop[indx].data['u'].reshape(1, -1)
            x0_new_final_pop[indx, :, :] = res.pop[indx].data['x0_new']
        np.savetxt(os.path.join(save_folder, 'stress_pop_max_gen'), stress_final_pop)
        np.savetxt(os.path.join(save_folder, 'strain_pop_max_gen'), strain_final_pop)
        np.savetxt(os.path.join(save_folder, 'u_pop_max_gen'), u_final_pop)
        np.save(os.path.join(save_folder, 'x0_new_pop_max_gen'), x0_new_final_pop)

        # Save pymoo result object
        pickle.dump(res, open(os.path.join(save_folder, 'pymoo_result.pickle'), 'wb'))

        t1 = time.time()
        total_execution_time = t1 - t0
        print(f"Total execution time {total_execution_time}")  # Seconds elapsed
        logging.info(f"Total execution time {total_execution_time}")

        # Plot results
        plt.scatter(res.F[:, 0], res.F[:, 1])
        plt.xlabel("Weight (kg)")
        plt.ylabel("Compliance (m/N)")
        plt.savefig(os.path.join(save_folder, 'pf.png'))
    plt.show()
