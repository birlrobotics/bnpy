'''
MoCap124.py

Dataset generated by motion capture of humans performing various exercises.
'''

import numpy as np
import readline
import os
import scipy.io

from bnpy.data import GroupXData

datasetdir = os.path.sep.join(
    os.path.abspath(__file__).split(os.path.sep)[:-1])

if not os.path.isdir(datasetdir):
    raise ValueError('CANNOT FIND DATASET DIRECTORY:\n' + datasetdir)

matfilepath = os.path.join(datasetdir, 'rawData', 'MoCap124.mat')
if not os.path.isfile(matfilepath):
    raise ValueError('CANNOT FIND DATASET MAT FILE:\n' + matfilepath)


def get_data(**kwargs):
    Data = GroupXData.read_from_mat(matfilepath)
    Data.summary = get_data_info()
    Data.name = get_short_name()
    return Data


def get_data_info():
    return '124 sequences of motion capture data from mocap.cs.cmu.edu'


def get_short_name():
    return 'MoCap124'

# How to make MAT file
###########################################################
# Exact commands to execute in python interpreter (by hand)
# to create a MAT file from the raw data distributed by NPBayesHMM toolbox
# ---------
# >> dpath = '/path/to/git/NPBayesHMM/data/mocap6/'
# >> SaveVars = loadFromPlainTextFiles(dpath)
# >> scipy.io.savemat('/path/to/git/bnpy-dev/datasets/MoCap6.mat', SaveVars)

# Reproducibility Notes
# ---------
# Mimics the following files in NPBayesHMM repository
# * readSeqDataFromPlainText.m
# * ARSeqData.m (specifically 'addData' method)


def CreateBNPYMatFile(origmatfile='/tmp/MoCapSensorData_R1_nCh12_W12.mat',
                      outmatfile='MoCap124.mat'):
    Vars = CreateBNPYDataDict(origmatfile)
    scipy.io.savemat(outmatfile, Vars, oned_as='row')


def CreateBNPYDataDict(origmatfile='/tmp/MoCapSensorData_R1_nCh12_W12.mat',
                       ):
    ''' Load old data used in NIPS 2012 paper into new bnpy data format

       Returns
       --------
       DataDict : dict with fields
       * X : data matrix
       * Xprev : matrix of previous observations
       * seqNames : list of strings, one per sequence
       * doc_range : ptr to where each seq stops and starts
    '''
    Vars = scipy.io.loadmat(origmatfile)
    Vars = Vars['data_struct']
    N = 124

    # Define the doc_range indices
    # indicate where each seq starts and stops
    # Remember that we need to discard "start observation" for each sequence
    # to allow autoregressive likelihoods
    T_all = 0
    doc_range = np.zeros(N + 1, dtype=np.int32)
    for n in xrange(N):
        T_n = np.squeeze(Vars['T'][0, n]) - 1  # len of seq n
        doc_range[n + 1] = doc_range[n] + T_n
    T_all = doc_range[-1]

    X = np.zeros((T_all, 12))
    Xprev = np.zeros_like(X)
    for n in xrange(N):
        X_n = np.asarray(Vars['obs'][0, n], dtype=np.float)
        start = doc_range[n]
        stop = doc_range[n + 1]
        X[start:stop] = X_n[1:].copy()
        Xprev[start:stop] = X_n[:-1].copy()

    # Track sequence names
    fileNames = list()
    for n in xrange(N):
        fileNames.append(str(np.squeeze(Vars['fileName'][0, n])))

    # Double check that expected relation between X and Xprev holds
    for seqID in xrange(doc_range.size - 1):
        start = doc_range[seqID]
        stop = doc_range[seqID + 1]
        assert np.allclose(X[start:stop - 1], Xprev[start + 1:stop])

    return dict(X=X, Xprev=Xprev,
                fileNames=fileNames,
                doc_range=doc_range,
                )
