#!/usr/bin/env python
import os
import pandas as pd
import numpy as np
import bnpy
import time
from sklearn.externals import joblib
from math import (
    log,
    exp
)
from matplotlib import pyplot as plt
import ipdb

def make_trials_of_each_state_the_same_length(trials_group_by_folder_name):
    # may implement DTW in the future...
    # for now we just align trials with the shortest trial of each state

    one_trial_data_group_by_state = trials_group_by_folder_name.itervalues().next()
    state_amount = len(one_trial_data_group_by_state)

    for state_no in range(1, state_amount+1):

        min_length = None
        for trial_name in trials_group_by_folder_name:
            # remember that the actual data is a numpy matrix
            # so we use *.shape[0] to get the length
            now_length = trials_group_by_folder_name[trial_name][state_no].shape[0]
            if min_length is None or now_length < min_length:
                min_length = now_length

        # align all trials in this state to min_length
        for trial_name in trials_group_by_folder_name:
            trials_group_by_folder_name[trial_name][state_no] = trials_group_by_folder_name[trial_name][state_no][:min_length, :]

    return trials_group_by_folder_name

def assess_threshold_and_decide(mean_of_log_curve, std_of_log_curve,
                                np_matrix_traj_by_time,
                                curve_owner, state_no,
                                figure_save_path):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # plot log curves of all trials
    for row_no in range(np_matrix_traj_by_time.shape[0]):
        trial_name = curve_owner[row_no]
        if row_no == 0:
            ax.plot(np_matrix_traj_by_time[row_no].tolist()[0], linestyle="dashed", color='gray', label='trials')
        else:
            ax.plot(np_matrix_traj_by_time[row_no].tolist()[0], linestyle="dashed", color='gray')
    # plot mean-c*std log curve
    for c in np.arange(0, 20, 2):
        ax.plot((mean_of_log_curve - c*std_of_log_curve).tolist()[0], label="mean-%s*std"%(c,), linestyle='solid')

    # decide c in an interactive way
    print "default c = 10 to visualize mean-c*std or enter ok to use this c as final threshold:"
    c = 10 # this is default

    ax.plot((mean_of_log_curve - c * std_of_log_curve).tolist()[0], label="mu-%s*std" % (c,), linestyle='dotted')
    ax.legend()
    title = 'state %s use threshold with c=%s' % (state_no, c)
    ax.set_title(title)
    if not os.path.isdir(figure_save_path):
        os.makedirs(figure_save_path)
    fig.savefig(os.path.join(figure_save_path + '/', title + ".eps"), format="eps")
    return mean_of_log_curve - c * std_of_log_curve

def run(model_save_path, 
    figure_save_path,
    trials_group_by_folder_name,
    ):

    trials_group_by_folder_name = make_trials_of_each_state_the_same_length(trials_group_by_folder_name)

    one_trial_data_group_by_state = trials_group_by_folder_name.itervalues().next()
    state_amount = len(one_trial_data_group_by_state)

    model_group_by_state = {}

    start_time = time.time()
    for state_no in range(1, state_amount+1):
        # -hdphmm
        model_group_by_state[state_no] = bnpy.load_model_at_prefix(model_save_path + '/' + str(state_no), 'Best')
    expected_log = []
    std_of_log   = []
    threshold    = []
    for state_no in range(1, state_amount+1):
        all_log_curves_of_this_state = []
        curve_owner = []
        for trial_name in trials_group_by_folder_name:
            curve_owner.append(trial_name)
            one_log_curve_of_this_state = [] 
            for time_step in range(0, len(trials_group_by_folder_name[trial_name][state_no]), 1):
                if not np.mod(time_step,50):
                    print str(state_no) + '-' + str(time_step) + '/' + str(len(trials_group_by_folder_name[trial_name][state_no]))
                #-start--hdp-hmm
                X         = trials_group_by_folder_name[trial_name][state_no][:time_step + 1]
                Xprev     = trials_group_by_folder_name[trial_name][state_no][:time_step + 1]
                doc_range = [0, time_step + 1]
                dataset   = bnpy.data.GroupXData(X, doc_range, time_step + 1, Xprev)
                LP        = model_group_by_state[state_no].calc_local_params(dataset)
                SS        = model_group_by_state[state_no].get_global_suff_stats(dataset, LP)
                log_probability = model_group_by_state[state_no].obsModel.calcMargLik(SS)
                # -end--hdp-hmm

                one_log_curve_of_this_state.append(log_probability)
            all_log_curves_of_this_state.append(one_log_curve_of_this_state)

        # use np matrix to facilitate the computation of mean curve and std 
        np_matrix_traj_by_time = np.matrix(all_log_curves_of_this_state)
        mean_of_log_curve      = np_matrix_traj_by_time.mean(0)
        std_of_log_curve       = np_matrix_traj_by_time.std(0)

        decided_threshold_log_curve = assess_threshold_and_decide(mean_of_log_curve,
                                                                  std_of_log_curve,
                                                                  np_matrix_traj_by_time,
                                                                  curve_owner,
                                                                  state_no,
                                                                  figure_save_path)
        expected_log.append(mean_of_log_curve.tolist()[0])
        threshold.append(decided_threshold_log_curve.tolist()[0])
        std_of_log.append(std_of_log_curve.tolist()[0])
        print time.time() - start_time

    if not os.path.isdir(model_save_path ):
        os.makedirs(model_save_path )
        
    joblib.dump(expected_log, model_save_path + '/expected_log.pkl')
    joblib.dump(threshold,    model_save_path + '/threshold.pkl')
    joblib.dump(std_of_log,   model_save_path + '/std_of_log.pkl')
