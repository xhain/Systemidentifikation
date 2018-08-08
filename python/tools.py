# -*- coding: utf-8 -*-
"""
Created on Mo Aug 06 21:42:12 2018

@author: Maximilian Weber

Adaptive Filter, SoSe 2018, Prof. Sikora
File: Toolbox
"""

import os
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt


#####
def addNoise(x,variance):
    """
    Adds noise to a signal x with a desired variance
    
    """
    
    # Calculate standard deviation sigma from input variance
    sigma = np.sqrt(variance)
    
    # Get length of input vector
    xLen = x.shape[1]
    
    # Generate gaussian noise with mean = 0, length of x 
    # and the standard deviation sigma
    noise = np.random.normal(0,sigma,xLen)
    
    # Add noise to input signal
    x_noise = x + noise
    return x_noise



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
def plotvecs(X,title='No Title',style='lin',xLim=400):
    """
    Plot Vectors from Array
    
    """
    plt.figure(figsize=(12, 4))
    plt.title(title)
    for x in X:
        if style == 'log':
            plt.semilogy(x)
        elif style == 'lin':
            plt.plot(x)
        plt.xlim(0,xLim)

    
#####
def plotdb(X,title='No Title',xLim=400):
    """
    Plot dB scale from vector X
    
    """
    x = 20 * np.log10(X)
    plt.figure(figsize=(12, 4))
    plt.title(title)
    plt.plot(x)
    