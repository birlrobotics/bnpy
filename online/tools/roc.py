#!/usr/bin/env python
import sys
import os
import pandas as pd
import numpy as np
from hmmlearn.hmm import *
from sklearn.externals import joblib
import ipdb
import time
from math import (
    log,
    exp
)
from matplotlib import pyplot as plt
from sklearn.preprocessing import (
    scale,
    normalize
)



def matplot_list(list_data,
                 figure_index,
                 title,
                 label_string,
                 save_path,
                 save=False,
                 linewidth='3.0',
                 fontsize= 50,
                 xaxis_interval=0.005,
                 xlabel= 'time',
                 ylabel = 'log likelihood'):
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
    plt.xlabel(xlabel,fontsize=fontsize)
  

    plt.ylabel(ylabel,fontsize=fontsize)

    plt.xticks( fontsize = 50)
    plt.yticks( fontsize = 50)
    
    for data in list_data:
        i = i + 1
        index = np.asarray(data).shape
        O = (np.arange(index[0])*xaxis_interval).tolist()
        if label_string[i-1] == 'threshold':
            plt.plot(O, data, label=label_string[i-1],linewidth=3, linestyle = '--', mfc ="grey")
        else:
            plt.plot(O, data, label=label_string[i-1],linewidth=linewidth)
    plt.legend(loc='best', frameon=True, fontsize=fontsize)

    plt.title(title, fontsize=fontsize)

    #plt.annotate('State=4 Sub_State='+str(n_state)+' GaussianHMM_cov='+covariance_type_string,
    #         xy=(0, -5000), xycoords='data',
    #         xytext=(+10, +30), textcoords='offset points', fontsize=fontsize,
    #         arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-.25"))

    if save:
        plt.savefig(save_path+title+".eps", format="eps")







    
def main():

    x =[0,  0,  0, 5.5,   9, 10, 20,24.5,  36, 45.5, 60,  80,  100]
    y =[0, 18, 30,  40,  55, 60, 70,  80,  86, 89.5, 91.5, 92, 100]


    x_2 =[0, 0, 0, 3.13,6.25,9.38,18.75,31.25, 40.63, 62.5, 78.13, 81.25, 84.38, 100]
    y_2 =[2.5, 25, 50, 55, 70, 77.5, 77.5, 90, 92.5, 95, 95, 95, 95, 100  ]

    x_3 =[0,0,0,3.13,6.25,9.38,18.75,21.25,40.63,62.5,78.13,81.25,84.38]
    y_3 =[2.5,45,85,88,87,87.75,87.75,90,92.5,95,95,95,95]

    #plt.scatter(x, y, s=20, c='red')
    plt.plot(x,y, 'o-',label="HMM",linewidth=1,color='red',markersize=6)
    #plt.plot(x_2,y_2, 'o-',label="sHDP-HMM",linewidth=1,color='green',markersize=6)
    #plt.plot(x_3,y_3, 'o-',label="sHDP-VAR-HMM",linewidth=1,markersize=6)
    


    
    #plt.plot(x,y, 'o', mfc ="red",makersize = '3.0')
    x1 = [0,100]
    y1 = [0,100]
    plt.xlabel("False Positive Rate(%)")

    plt.xlim([0,100])
    plt.ylim([0,100])

    plt.ylabel("True Positive Rate()")
    plt.plot(x1,y1,linewidth=1, linestyle = '--', color ="grey")
    plt.legend(loc='best', frameon=True)
    plt.title("ROC for Anomaly Detection")
    #plt.show()

    plt.savefig("roc.eps", format="eps")
    return 0

if __name__ == '__main__':
    sys.exit(main())
