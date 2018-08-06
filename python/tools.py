# -*- coding: utf-8 -*-
"""
Created on Mo Aug 06 21:42:12 2018

@author: Maximilian Weber

Adaptive Filter, SoSe 2018, Prof. Sikora
File: Toolbox
"""

import os
import scipy.io as sio
import matplotlib.pyplot as plt

#####
def importmat(filepath):
    """
    Imports *.mat files to workspace
    
    """
    
    importMats = {}
    importFileNames = []
    
    print('*** Importing files from directory: '+filepath)
    for file in os.listdir(filepath):
        importFileName = file.split('.')[0]
        print('* importing: '+file)
        mat = sio.loadmat(filepath+file)
        importMats[importFileName] = mat
        importFileNames.append(importFileName)
    print('*** '+str(len(importFileNames))+' files imported')
    return importMats, importFileNames


#####
def plotvecs(X,title,style):
    """
    Plot Vectors from Array
    
    """
    plt.figure()
    plt.title(title)
    for x in X:
        if style == 'log':
            plt.semilogy(x)
        elif style == 'lin':
            plt.plot(x)