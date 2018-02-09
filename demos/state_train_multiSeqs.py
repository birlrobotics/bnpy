"""
@Hongmin Wu
==============================================
Training the different models by multiple sequential trials
==============================================

How to train mixtures and HMMs with various observation models on the same dataset.

"""
import bnpy
import numpy as np
import os
from matplotlib import pylab
import glob
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category= DeprecationWarning)

SMALL_FIG_SIZE = (2.5, 2.5)
FIG_SIZE = (5, 5)
pylab.rcParams['figure.figsize'] = FIG_SIZE
pylab.close('all')

dataPath     = '/media/vmrguser/DATA/Homlx/DATA_HIRO_SA/REAL_HIRO_ONE_SA_SUCCESS'
file_id      = ['02', '03', '04', '05', '06','07', '08', '09', '10', '11', '12', '13', '14', '15', '16']
nLap         = 500
time_step    = 0.005
dataType     = ['R_Torques.dat'] #'R_Angles.dat','R_CartPos.dat' tuples can not edited
STATE        = ['APPROACH', 'ROTATION', 'INSERTION', 'MATING']

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
    stop  = train_dataset.doc_range[seq_id + 1]
    for dim in xrange(12):
        X_seq = train_dataset.X[start:stop]
        pylab.plot(X_seq[:, dim], '.-')
    pylab.title(STATE[iState])
    pylab.xlabel('time')
    pylab.ylabel('angle')
    pylab.tight_layout()

#generate the dataset by the hiro data
########################################################################################################################
#training trial
########################################################################################################################
for iState in range(len(STATE)):
    raw_data  = np.empty((0, 6),float)
    doc_range = list([0])
    TrueZ     = list()
    for idx in range(len(file_id)):
        folder_name, _data, Rstate = load_one_trial(dataPath, file_id[idx])
        _data     = _data[int(Rstate[iState]): int(Rstate[iState + 1]), :]
        raw_data  = np.vstack((raw_data, _data))
        doc_range.append(len(_data))
        TrueZ.append(len(_data))
    X             = raw_data[:-1,:]
    Xprev         = raw_data[1::,:] #only for the first-order Autoregressive
    doc_range     = np.cumsum(np.asarray(doc_range))
    doc_range[-1] = doc_range[-1] - 1
    np.savez(os.path.join(bnpy.DATASET_PATH, 'hiro/multiSeq_dataset.npz'), **dict(X=X, Xprev=Xprev, doc_range=doc_range, TrueZ=TrueZ))
    train_dataset = bnpy.data.GroupXData.read_npz(os.path.join(bnpy.DATASET_PATH, 'hiro/multiSeq_dataset.npz'))
    ########################################################################################################################

    # Visualization of the first/second sequence
    # -----------------------------------
    #show_single_sequence(0) # first
    #show_single_sequence(1) #second

    ###############################################################################
    #
    # Setup: hyperparameters
    # ----------------------------------------------------------
    alpha = 0.5
    gamma = 5.0 #default 5.0
    sF    = 1.0
    K     = 50
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
    modelPath = 'saveModels/DPMixtureModel+DiagGauss+memoVB/' + STATE[iState]
    savePath  = os.path.join(bnpy.ROOT_PATH, modelPath)
    if not os.path.isdir(savePath):
        os.makedirs(savePath)
    bnpy.save_model(mixdiag_trained_model, savePath, 'Best')

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
    #save the trained model
    modelPath = 'saveModels/HDPHMM+DiagGauss+memoVB/' + STATE[iState]
    savePath  = os.path.join(bnpy.ROOT_PATH, modelPath)
    if not os.path.isdir(savePath):
        os.makedirs(savePath)
    bnpy.save_model(hmmdiag_trained_model, savePath, 'Best')
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
    #save the trained model
    modelPath = 'saveModels/HDPHMM+Gauss+memoVB/' + STATE[iState]
    savePath  = os.path.join(bnpy.ROOT_PATH, modelPath)
    if not os.path.isdir(savePath):
        os.makedirs(savePath)
    bnpy.save_model(hmmfull_trained_model, savePath, 'Best')

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
    #save the trained model
    modelPath = 'saveModels/HDPHMM+AutoRegGauss+memoVB/' + STATE[iState]
    savePath  = os.path.join(bnpy.ROOT_PATH, modelPath)
    if not os.path.isdir(savePath):
        os.makedirs(savePath)
    bnpy.save_model(hmmar_trained_model, savePath, 'Best')

'''
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
pylab.draw()
pylab.tight_layout()
pylab.show()

'''
