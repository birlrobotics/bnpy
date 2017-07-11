#!/usr/bin/env python
import os
import numpy as np
import hmmlearn.hmm
import ipdb
import bnpy
from sklearn.externals import joblib
from sklearn.preprocessing import (
    scale,
    normalize
)

def run(model_save_path, 
    n_state,
    n_iteration,
    alloModel,
    obsModel,
    varMethod,
    trials_group_by_folder_name):

    list_of_trials = trials_group_by_folder_name.values() 

    if not os.path.isdir(model_save_path):
        os.makedirs(model_save_path)

    one_trial_data_group_by_state = list_of_trials[0]
    state_amount = len(one_trial_data_group_by_state)

    start_prob = np.zeros(state_amount)
    start_prob[0] = 1

    training_data_group_by_state = {}
    training_length_array_group_by_state = {}

    for state_no in range(1, state_amount+1):
        length_array = []
        for trial_no in range(len(list_of_trials)):
            length_array.append(list_of_trials[trial_no][state_no].shape[0])
            if trial_no == 0:
                data_tempt = list_of_trials[trial_no][state_no]
            else:
                data_tempt = np.concatenate((data_tempt,list_of_trials[trial_no][state_no]),axis = 0)
        training_data_group_by_state[state_no]         = data_tempt
        training_length_array_group_by_state[state_no] = length_array

    model_group_by_state = {}
    for state_no in range(1, state_amount+1):
        # -start--hdp-hmm
        X          = training_data_group_by_state[state_no]
        Xprev      = training_data_group_by_state[state_no] #only for the first autoregressive
        doc_range  = list([0])
        doc_range += (np.cumsum(training_length_array_group_by_state[state_no]).tolist())
        # TrueZ     = training_length_array_group_by_state[state_no]
        dataset    = bnpy.data.GroupXData(X, doc_range, None, Xprev)

        # -set the hyperparameters
        alpha      = 0.5
        gamma      = 5.0
        sF         = 1.0
        model, model_info = bnpy.run(dataset,
                                     alloModel,
                                     obsModel,
                                     varMethod,
                                     output_path = '/home/Homls/npBayesHMM/python/bnpy/results',
                                     nLap        = n_iteration,
                                     nTask       = 1,
                                     nBatch      = 1,
                                     convergethr = 0.00001,
                                     alpha       = alpha,
                                     gamma       = gamma,
                                     sF          = sF,
                                     ECovMat     = 'eye',
                                     K           = n_state,
                                     initname    = 'randexamples',
                                    )
        # ipdb.set_trace()
        model_group_by_state[state_no] = model

        # save the models
        if not os.path.isdir(model_save_path + '/' + str(state_no)):
            os.makedirs(model_save_path + '/' + str(state_no))
        bnpy.save_model(model, model_save_path + '/' + str(state_no), 'Best')
    # -end--hdp-hmm
