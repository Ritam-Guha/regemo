import logging
import os
import pickle
from shutil import copyfile

import h5py
import numpy as np
import datetime

# from optimal_power_flow.run_mopf import results_dir
results_parent_dir = 'results'
# results_dir = os.path.join(results_parent_dir, f'{ts}')
time_now = datetime.datetime.now()
results_dir = os.path.join(results_parent_dir, f'{time_now.strftime("%Y%m%d_%H%M%S")}')


def record_state(algorithm):
    global results_dir
    logging.getLogger().info(f"Gen {algorithm.n_gen}")
    rank_pop = algorithm.pop.get('rank')
    x_pop = algorithm.pop.get('X')
    f_pop = algorithm.pop.get('F')
    rank_pop[rank_pop == None] = -1
    rep_time_pop = algorithm.pop.get('rep_time')
    rep_indx_pop = algorithm.pop.get('rep_indx')

    x_pf = x_pop[rank_pop == 0]
    pf = f_pop[rank_pop == 0]
    x_points_of_interest = np.copy(x_pf)
    # print(len(pf))
    # if len(pf) > 10:
    #     knee = get_knee(pf, epsilon=0.125)
    #     min_x_knee = np.min(x_pf[knee, 0])
    #     max_x_knee = np.max(x_pf[knee, 0])
    #     if min_x_knee != max_x_knee:
    #         mean_x_knee = max_x_knee - min_x_knee
    #     else:
    #         mean_x_knee = min_x_knee
    #     x_points_of_interest = x_pf[(x_pf[:, 0] >= (mean_x_knee - 0.15)) & (x_pf[:, 0] <= (mean_x_knee + 0.15)), :]
    #     if len(x_points_of_interest) <= 1:
    #         x_points_of_interest = np.copy(x_pf)

    g_pop = np.array([])
    cv_pop = np.array([])
    if algorithm.problem.n_constr > 0:
        g_pop = algorithm.pop.get('G')
        cv_pop = algorithm.pop.get('CV')

    # Percentage of pop in ND set serves as a general confidence of rules learned
    algorithm.problem.percent_pf = pf.shape[0] / algorithm.pop_size
    # if algorithm.problem.percent_pf >= 0.5:
    innov_info_available = False
    innov = None
    if pf.shape[0] > 10:
        # If atleast 10 gens have passed and just before repair occurs
        if (algorithm.n_gen >= 5
                and (algorithm.n_gen + 1) % algorithm.problem.repair_interval == 0
                and x_points_of_interest.shape[0] >= 1):
            innov_info_available = True
            
            

    if algorithm.n_gen % 100 == 0:
        # print("Save state disabled.")
        print("Saving state")
        with open(os.path.join(results_dir, 'state.pkl'), 'wb') as f:
            pickle.dump(algorithm, f)

    optim_state_hdf_file = os.path.join(results_dir, 'optim_state.hdf5')
    with h5py.File(optim_state_hdf_file, 'a') as hf:
        hf.attrs['obj_label'] = algorithm.problem.obj_label
        hf.attrs['current_gen'] = algorithm.n_gen
        hf.attrs['xl'] = algorithm.problem.xl
        hf.attrs['xu'] = algorithm.problem.xu
        hf.attrs['n_obj'] = algorithm.problem.n_obj
        hf.attrs['n_constr'] = algorithm.problem.n_constr
        if 'innov_info_latest_gen' not in hf.attrs.keys():
            hf.attrs['innov_info_latest_gen'] = -1
        if 'total_rep_time' not in hf.attrs:
            hf.attrs['total_rep_time'] = 0
        if rep_time_pop[0] is not None:
            hf.attrs['total_rep_time'] += rep_time_pop[0]
            logging.info(f"Total repair time = {hf.attrs['total_rep_time']}")
        # if innov_info_available and innov is not None:
            # hf.attrs['power_law_error_max'] = innov.power_law_error_max
        # if 'power_law_error_max' not in hf.attrs.keys():
        #     hf.attrs['power_law_error_max'] = np.zeros(1)  # np.zeros([algorithm.problem.n_var, algorithm.problem.n_var])
        hf.attrs['ignore_vars'] = algorithm.problem.ignore_vars

        g1 = hf.create_group(f'gen{algorithm.n_gen}')

        # Basic population data
        g1.create_dataset('X', data=x_pop)
        g1.create_dataset('F', data=f_pop)
        g1.create_dataset('rank', data=rank_pop.astype(int))
        g1.create_dataset('G', data=g_pop)
        g1.create_dataset('CV', data=cv_pop)
        g1.create_dataset('rep_time', data=rep_time_pop[0])
        g1.create_dataset('rep_indx', data=rep_indx_pop)

        # Innovization information
        if innov_info_available:
            hf.attrs['innov_info_latest_gen'] = algorithm.n_gen
            g1.create_dataset('percent_pf', data=algorithm.problem.percent_pf)
            g1.create_dataset('var_groups', data=algorithm.problem.var_groups)
            g1.create_dataset('var_group_score', data=algorithm.problem.var_group_score)
            g1.create_dataset('corr', data=algorithm.problem.corr)
            g1.create_dataset('rel_type', data=algorithm.problem.rel_type)

    if os.path.exists(optim_state_hdf_file):
        copyfile(optim_state_hdf_file, optim_state_hdf_file + ".bak")
