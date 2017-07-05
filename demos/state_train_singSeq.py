"""
==============================================
Comparing models for sequential data
==============================================

How to train mixtures and HMMs with various observation models on the same dataset.

"""
import bnpy
import numpy as np
import os

from matplotlib import pylab
from sklearn import preprocessing
import glob
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category= DeprecationWarning)
import seaborn as sns

SMALL_FIG_SIZE = (2.5, 2.5)
FIG_SIZE = (5, 5)
pylab.rcParams['figure.figsize'] = FIG_SIZE
pylab.close('all')

nLap         = 500
time_step    = 0.005
dataType     = ['R_Torques.dat','R_Angles.dat'] #'R_Angles.dat','R_CartPos.dat' tuples can not edited
STATE        = ['APPROACH', 'ROTATION', 'INSERTION', 'MATING']
iState       = 0

###############################################################################
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

###############################################################################
#
# Setup: Function to make a simple plot of the raw data
# -----------------------------------------------------
def show_single_sequence(seq_id):
    start = train_dataset.doc_range[seq_id]
    stop = train_dataset.doc_range[seq_id + 1]
    for dim in xrange(12):
        X_seq = train_dataset.X[start:stop]
        pylab.plot(X_seq[:, dim], '.-')
    pylab.title(STATE[iState])
    pylab.xlabel('time')
    pylab.ylabel('Multi-modal')
    pylab.tight_layout()

dataPath               = '/home/birl/npBayesHMM/HIRO_SA_DATA/REAL_HIRO_ONE_SA_SUCCESS'
file_id                = '02'
folder_name, raw_data, Rstate = load_one_trial(dataPath, file_id)

#generate the dataset by the hiro data
########################################################################################################################
#training trial
########################################################################################################################
train_data  = raw_data[int(Rstate[iState ]) : int(Rstate[iState + 1]), :]
X         = train_data[:-1,:]
Xprev     = train_data[1::,:] #only for the first-order Autoregressive
doc_range = [0, train_data.shape[0] - 1]
TrueZ     = np.asarray([train_data.shape[0] - 1])
np.savez('/home/birl/npBayesHMM/python/bnpy/bnpy/datasets/hiro/train_dataset.npz', **dict(X=X, Xprev=Xprev, doc_range=doc_range, TrueZ=TrueZ))
dataset_path  = os.path.join(bnpy.DATASET_PATH, 'hiro')
train_dataset = bnpy.data.GroupXData.read_npz( os.path.join(dataset_path, 'train_dataset.npz'))
########################################################################################################################

#testing trial
########################################################################################################################
test_data  = raw_data[int(Rstate[iState]) : int(Rstate[iState + 1]), :]
X         = test_data[:-1,:]
Xprev     = test_data[1::,:] #only for the first-order Autoregressive
doc_range = [0, test_data.shape[0] - 1]
TrueZ     = np.asarray([test_data.shape[0] - 1])
np.savez('/home/birl/npBayesHMM/python/bnpy/bnpy/datasets/hiro/test_dataset.npz', **dict(X=X, Xprev=Xprev, doc_range=doc_range, TrueZ=TrueZ))
dataset_path  = os.path.join(bnpy.DATASET_PATH, 'hiro')
test_dataset = bnpy.data.GroupXData.read_npz( os.path.join(dataset_path, 'test_dataset.npz'))
########################################################################################################################

# Visualization of the first/second sequence
# -----------------------------------
show_single_sequence(0) # first
#show_single_sequence(1) #second


###############################################################################
#
# Setup: hyperparameters
# ----------------------------------------------------------
alpha = 0.5
gamma = 5.0
sF    = 1.0
K     = 20

###############################################################################
#
# DP mixture with *DiagGauss* observation model
# ---------------------------------------------
mixdiag_trained_model, mixdiag_info_dict = bnpy.run(
    train_dataset, 'DPMixtureModel', 'DiagGauss', 'memoVB',
    output_path='/tmp/mocap6/showcase-K=20-model=DP+DiagGauss-ECovMat=1*eye/',
    nLap=nLap, nTask=1, nBatch=1, convergeThr=0.0001,
    alpha=alpha, gamma=gamma, sF=sF, ECovMat='eye',
    K=K, initname='randexamples',
    )
#save the trained model
# bnpy.save_model(mixdiag_trained_model, '/home/birl/npBayesHMM/python/bnpy/saveModels', 'Best')
#load the trained model
# loadModel = bnpy.load_model_at_prefix('/home/birl/npBayesHMM/python/bnpy/saveModels',   'Best')

###############################################################################
#
# HDP-HMM with *DiagGauss* observation model
# -------------------------------------------
#
# Assume diagonal covariances.
#
# Start with too many clusters (K=20)
hmmdiag_trained_model, hmmdiag_info_dict = bnpy.run(
    train_dataset, 'HDPHMM', 'DiagGauss', 'memoVB',
    output_path='/tmp/mocap6/showcase-K=20-model=HDPHMM+DiagGauss-ECovMat=1*eye/',
    nLap=nLap, nTask=1, nBatch=1, convergeThr=0.0001,
    alpha=alpha, gamma=gamma, sF=sF, ECovMat='eye',
    K=K, initname='randexamples',
    )


###############################################################################
#
# HDP-HMM with *Gauss* observation model
# --------------------------------------
#
# Assume full covariances.
#
# Start with too many clusters (K=20)
hmmfull_trained_model, hmmfull_info_dict = bnpy.run(
    train_dataset, 'HDPHMM', 'Gauss', 'memoVB',
    output_path='/tmp/mocap6/showcase-K=20-model=HDPHMM+Gauss-ECovMat=1*eye/',
    nLap=nLap, nTask=1, nBatch=1, convergeThr=0.0001,
    alpha=alpha, gamma=gamma, sF=sF, ECovMat='eye',
    K=K, initname='randexamples',
    )


###############################################################################
#
# HDP-HMM with *AutoRegGauss* observation model
# ----------------------------------------------
#
# Assume full covariances.
#
# Start with too many clusters (K=20)
hmmar_trained_model, hmmar_info_dict = bnpy.run(
    train_dataset, 'HDPHMM', 'AutoRegGauss', 'memoVB',
    output_path='/tmp/mocap6/showcase-K=20-model=HDPHMM+AutoRegGauss-ECovMat=1*eye/',
    nLap=nLap, nTask=1, nBatch=1, convergeThr=0.0001,
    alpha=alpha, gamma=gamma, sF=sF, ECovMat='eye',
    K=K, initname='randexamples',
    )

#calculate the log-likelihood of trained model
mixdiag_trained_model_LP = mixdiag_trained_model.calc_local_params(test_dataset)
hmmdiag_trained_model_LP = hmmdiag_trained_model.calc_local_params(test_dataset)
hmmfull_trained_model_LP = hmmfull_trained_model.calc_local_params(test_dataset)
hmmar_trained_model_LP   = hmmar_trained_model.calc_local_params(test_dataset)


pylab.figure()
pylab.title('Cumulative Log-likelihood')
# pylab.plot(mixdiag_trained_model_LP['E_log_soft_ev'])
pylab.plot(np.cumsum(np.amax(mixdiag_trained_model_LP['E_log_soft_ev'], axis=1)),'b-',label='mixdiag_trained_model')
pylab.plot(np.cumsum(np.amax(hmmdiag_trained_model_LP['E_log_soft_ev'], axis=1)),'k-',label='hmmdiag_trained_model')
pylab.plot(np.cumsum(np.amax(hmmfull_trained_model_LP['E_log_soft_ev'], axis=1)),'r-',label='hmmfull_trained_model')
pylab.plot(np.cumsum(np.amax(hmmar_trained_model_LP  ['E_log_soft_ev'], axis=1)),'c-',label='hmmar_trained_model')
pylab.legend(loc='upper right')
###############################################################################
#
# Compare loss function traces for all methods
# --------------------------------------------
#
pylab.figure()
pylab.title('Objective Loss Function')
pylab.plot(
    mixdiag_info_dict['lap_history'],
    mixdiag_info_dict['loss_history'], 'b.-',
    label='mix + diag gauss')

pylab.plot(
    hmmdiag_info_dict['lap_history'],
    hmmdiag_info_dict['loss_history'], 'k.-',
    label='hmm + diag gauss')

pylab.plot(
    hmmfull_info_dict['lap_history'],
    hmmfull_info_dict['loss_history'], 'r.-',
    label='hmm + full gauss')

pylab.plot(
    hmmar_info_dict['lap_history'],
    hmmar_info_dict['loss_history'], 'c.-',
    label='hmm + ar gauss')

pylab.legend(loc='upper right')
pylab.xlabel('num. laps')
pylab.ylabel('loss')
# pylab.xlim([4, 100]) # avoid early iterations
# pylab.ylim([2.4, 3.7]) # handpicked
pylab.draw()
pylab.tight_layout()
pylab.show()
