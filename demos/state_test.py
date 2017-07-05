"""
@Hongmin Wu
==============================================
Testing the trails
==============================================

"""
import bnpy
import numpy as np
import os
from matplotlib import pylab
import glob
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category= DeprecationWarning)

modelList    = ['DPMixtureModel+DiagGauss+memoVB',  # 0
                'HDPHMM+DiagGauss+memoVB',          # 1
                'HDPHMM+Gauss+memoVB',              # 2
                'HDPHMM + AutoRegGauss + memoVB']   # 3
STATE        = ['APPROACH',
                'ROTATION',
                'INSERTION',
                'MATING']
model_ID     = 2
dataPath     = '/home/birl/npBayesHMM/HIRO_SA_DATA/REAL_HIRO_ONE_SA_SUCCESS'
file_id      = ['02', '03','04', '05', '06', '07', '08', '09', '10', '11']
dataType     = ['R_Torques.dat','R_Angles.dat'] #'R_Angles.dat','R_CartPos.dat'
COLOR        = ['r', 'g', 'b', 'k']
COLOR_SYMBOL = ['rs', 'gs', 'bs', 'ks']
time_step    = 0.005

def load_one_trial(dataPath,id):
    sensor            = pd.DataFrame()
    Rstate            = pd.DataFrame()
    for folders in glob.glob(os.path.join(dataPath, "*" + id)):
        for dat_file in os.listdir(folders):
            if dat_file in dataType:
                raw_data  = pd.read_csv(folders +'/' + dat_file, sep='\s+', header=None, skiprows=1, usecols = range(1,7))
                sensor    = raw_data.transpose().append(sensor) # transpose to [ndim x length]
            elif dat_file == 'R_State.dat':
                Rstate    = pd.read_csv(folders + '/' + dat_file, sep='\s+', header=None, skiprows=1)
                Rstate    = Rstate.values / time_step
                Rstate    = np.insert(Rstate, 0, 0, axis= 0)
                Rstate    = np.append(Rstate, sensor.shape[1])
    return  folders, sensor.transpose().values, Rstate

'''
################################################
# the same testing state w.r.t trained different state
# calculate the log-likelihood by the evolving observation
################################################
#
# #load the testing data for given state = iState
folder_name, _data, Rstate = load_one_trial(dataPath, file_id[0])
test_data  = _data[int(Rstate[iState]) : int(Rstate[iState + 1]), :]

#load the models of different state
state_models = []
for n in range(len(STATE)):
    # modelPath = 'saveModels/HDPHMM+DiagGauss+memoVB/' + STATE[n]
    modelPath = 'saveModels/HDPHMM+Gauss+memoVB/' + STATE[n]
    # modelPath = 'saveModels/HDPHMM+AutoRegGauss+memoVB/' + STATE[n]
    loadPath = os.path.join(bnpy.ROOT_PATH, modelPath)
    if not os.path.isdir(loadPath):
        print 'Sorry, you had not trained this model before'
        os._exit(0)
    hmmar_trained_model = bnpy.load_model_at_prefix(loadPath, 'Best')
    state_models.append(hmmar_trained_model)

pylab.figure()
pylab.title('State Classification ' + STATE[iState])
for m in range(len(STATE)):
    state_log = []
    for obs in range(1, len(test_data), 1):
        print obs
        X     = test_data[:obs,:]
        Xprev = test_data[:obs,:]  # only for the first-order Autoregressive
        doc_range = [0, obs]
        TrueZ = np.asarray([obs])
        np.savez(os.path.join(bnpy.DATASET_PATH, 'hiro/test_dataset.npz'),
                 **dict(X=X, Xprev=Xprev, doc_range=doc_range, TrueZ=TrueZ))

        #calculate the log-likelihood of trained model
        test_dataset = bnpy.data.GroupXData.read_npz(os.path.join(bnpy.DATASET_PATH, 'hiro/test_dataset.npz'))
        hmmar_trained_model_LP   = state_models[m].calc_local_params(test_dataset)
        state_log.append(hmmar_trained_model_LP['evidence'])
    pylab.plot(state_log, COLOR[m], label = STATE[m])
pylab.xlabel('Time(ms)')
pylab.ylabel('evidence')
pylab.legend(loc='upper right')
pylab.show()
'''

################################################
#the same testing state w.r.t trained different state model
# calculate the log-likelihood by given the full observation
################################################
#
# #load the testing data for given state = iState
folder_name, _data, Rstate = load_one_trial(dataPath, file_id[0])

#load the models of different state
state_models = []
state_log = list()
for n in range(len(STATE)):
    #load testing data for each state
    state_data = _data[int(Rstate[n]): int(Rstate[n + 1]), :]
    X = state_data
    Xprev = state_data  # only for the first-order Autoregressive
    doc_range = [0, len(state_data)]
    TrueZ = np.asarray([len(state_data)])
    datasetName = 'hiro/' + 'state_' + str(n) + 'dataset.npz'
    np.savez(os.path.join(bnpy.DATASET_PATH, datasetName),
             **dict(X=X, Xprev=Xprev, doc_range=doc_range, TrueZ=TrueZ))

    #load the trained model
    modelPath = 'saveModels/' + modelList[model_ID] + '/' + STATE[n]
    loadPath = os.path.join(bnpy.ROOT_PATH, modelPath)
    if not os.path.isdir(loadPath):
        print 'Sorry, you had not trained this model before'
        os._exit(0)
    hmmar_trained_model = bnpy.load_model_at_prefix(loadPath, 'Best')
    state_models.append(hmmar_trained_model)

    state_log.append([]) # for storing the likelihood

for iState in range(len(STATE)):
    datasetName = 'hiro/' + 'state_' + str(iState) + 'dataset.npz'
    test_dataset = bnpy.data.GroupXData.read_npz(os.path.join(bnpy.DATASET_PATH, datasetName))
    for iModel in range(len(state_models)):
        LP        = state_models[iModel].obsModel.calc_local_params(test_dataset)
        state_log[iModel] += (np.cumsum(np.amax(LP['E_log_soft_ev'], axis=1))).tolist()

pylab.figure()
pylab.title('State Classification using ' + modelList[model_ID])
for i in range(len(STATE)):
    pylab.plot(state_log[i], COLOR[i], linewidth = 3.0, label = STATE[i])

pylab.xlabel('Time(ms)')
pylab.ylabel('evidence')
pylab.legend(loc='upper right')
pylab.show()