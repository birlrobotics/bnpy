#!/usr/bin/env python
import sys
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from hmmlearn.hmm import *
from sklearn.externals import joblib
import ipdb
from math import (
    log,
    exp
)
from sklearn.preprocessing import (
    scale,
    normalize
)

import time 

def matplot_list(list_data,
                 figure_index,
                 title,
                 label_string,
                 save=False,
                 linewidth='3.0'):
    # if you want to save, title is necessary as a save name.
    
    global n_state
    global covariance_type_string
    plt.figure(figure_index, figsize=(40,30), dpi=80)
    ax = plt.subplot(111)
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['bottom'].set_position(('data',0))
    plt.grid(True)
    i = 0
    for data in list_data:
        i = i + 1
        index = np.asarray(data).shape
        O = (np.arange(index[0])*0.01).tolist()
        plt.plot(O, data, label=label_string[i-1],linewidth=linewidth)
    plt.legend(loc='best', frameon=True)

    plt.title(title)

    plt.annotate('State=4 Sub_State='+str(n_state)+' GaussianHMM_cov='+covariance_type_string,
             xy=(0, 0), xycoords='data',
             xytext=(+10, +30), textcoords='offset points', fontsize=16,
             arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))

    if save:
        plt.savefig(title+".eps", format="eps")




def scaling(X):
    _index, _column = X.shape
    Data_scaled = []
    scale_length = 10
    for i in range(scale_length, _index-scale_length-2):
        scaled = scale(X[i-scale_length:i+scale_length + 1, :])
        Data_scaled.append(scaled[scale_length,:])

    scaled_array = np.asarray(Data_scaled)
    return scaled_array
    


def load_data(path, preprocessing_scaling=False, preprocessing_normalize=False, norm='l2'):
    """
       df.columns = u'time', u'.endpoint_state.header.seq',
       u'.endpoint_state.header.stamp.secs',
       u'.endpoint_state.header.stamp.nsecs',
       u'.endpoint_state.header.frame_id', u'.endpoint_state.pose.position.x',
       u'.endpoint_state.pose.position.y', u'.endpoint_state.pose.position.z',
       u'.endpoint_state.pose.orientation.x',
       u'.endpoint_state.pose.orientation.y',
       u'.endpoint_state.pose.orientation.z',
       u'.endpoint_state.pose.orientation.w',
       u'.endpoint_state.twist.linear.x', u'.endpoint_state.twist.linear.y',
       u'.endpoint_state.twist.linear.z', u'.endpoint_state.twist.angular.x',
       u'.endpoint_state.twist.angular.y', u'.endpoint_state.twist.angular.z',
       u'.endpoint_state.wrench.force.x', u'.endpoint_state.wrench.force.y',
       u'.endpoint_state.wrench.force.z', u'.endpoint_state.wrench.torque.x',
       u'.endpoint_state.wrench.torque.y', u'.endpoint_state.wrench.torque.z',
       u'.joint_state.header.seq', u'.joint_state.header.stamp.secs',
       u'.joint_state.header.stamp.nsecs', u'.joint_state.header.frame_id',
       u'.joint_state.name', u'.joint_state.position',
       u'.joint_state.velocity', u'.joint_state.effort',
       u'.wrench_stamped.header.seq', u'.wrench_stamped.header.stamp.secs',
       u'.wrench_stamped.header.stamp.nsecs',
       u'.wrench_stamped.header.frame_id', u'.wrench_stamped.wrench.force.x',
       u'.wrench_stamped.wrench.force.y', u'.wrench_stamped.wrench.force.z',
       u'.wrench_stamped.wrench.torque.x', u'.wrench_stamped.wrench.torque.y',
       u'.wrench_stamped.wrench.torque.z', u'.tag']
    """
    df = pd.read_csv(path+"/tag_multimodal.csv",sep=',')
    print "%s" %(path)

    df = df[[u'.wrench_stamped.wrench.force.x',
             u'.wrench_stamped.wrench.force.y',
             u'.wrench_stamped.wrench.force.z',
             u'.wrench_stamped.wrench.torque.x',
             u'.wrench_stamped.wrench.torque.y',
             u'.wrench_stamped.wrench.torque.z',
             u'.tag']]

    df = df[df.values[:,-2] !=0]
    
    Data = df.values

    index, columns = Data.shape

    time_data = []
    for i in range(index):
        time_data.append((i+1)*0.01)


    df['time'] = pd.Series(time_data)


    X_1 = df.values[df.values[:,-2] ==1]
    X_2 = df.values[df.values[:,-2] ==2]
    X_3 = df.values[df.values[:,-2] ==3]
    X_4 = df.values[df.values[:,-2] ==4]

    X_1_time = X_1[:,-1][-1]
    X_2_time = X_2[:,-1][-1]
    X_3_time = X_3[:,-1][-1]
    X_4_time = X_4[:,-1][-1]

    X_time = [[0],
              [X_1_time],
              [X_2_time],
              [X_3_time],
              [X_4_time]]
    X_time = np.array(X_time)

            

    df = df[[u'time',
             u'.wrench_stamped.wrench.force.x',
             u'.wrench_stamped.wrench.force.y',
             u'.wrench_stamped.wrench.force.z',
             u'.wrench_stamped.wrench.torque.x',
             u'.wrench_stamped.wrench.torque.y',
             u'.wrench_stamped.wrench.torque.z']]

    Data = df.values

    Data

    
        
    return Data, X_time



    
def main():

    ipdb.set_trace()

    index = ['01',
             '02',
             '03',
             '04',
             '05',
             '06',
             '07',
             '08',
             '09',
             '11',
             '12',
             '13',
             '14',
             '15',
             '16',
             '18',
             '19',
             '20',
             '21',
             '22',
             '23',
             '24']

    path = "/home/ben//ML_data/REAL_BAXTER_PICK_N_PLACE_5_18/success/"
    

    for i in range(24):
        Data,Time_Index = load_data(path=path+index[i])
        
        np.savetxt(path+index[i]+'/R_Torques.dat', Data, fmt='%.6f')
        np.savetxt(path+index[i]+'/R_State.dat', Time_Index, fmt='%.6f')
    

    return 0

if __name__ == '__main__':
    sys.exit(main())
