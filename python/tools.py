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
def plot(X,title='No Title',style='lin',xLim=400, xlab='Samples', ylab='MSE'):
    """
    Plot Vectors from Array
    
    """
    plt.figure(figsize=(12, 4))
    plt.title(title)
    for x in X:
        if style == 'log':
            plt.semilogy(x, 'b', linewidth=1)
        elif style == 'lin':
            plt.plot(x, 'b', linewidth=1)
        plt.xlim(0,xLim)
        plt.grid(True)
        plt.xlabel(xlab)
        plt.ylabel(ylab)

    
#####
def plotdb(X,title='No Title', xLim=400):
    """
    Plot dB scale from vector X
    
    """
    X = X / np.amax(np.abs(X))
    X = 20 * np.log10(np.abs(X))
    plt.figure(figsize=(12, 4))
    plt.title(title)
    plt.plot(X, linewidth=1)

    
    
#####
def hist(X, title):
    plt.figure(figsize=(12, 4))
    plt.title(title)
    plt.hist(X[0], facecolor='b')
    plt.grid(True)
    
#####
def errorPlot(E,W,plotLen=500):
    plt.figure(figsize=(12, 8))
    
    # Plot Error
    plt.subplot(211)
    plt.plot(E[0:plotLen], 'b', linewidth=1)
    plt.xlim(0,plotLen)
    plt.grid(True)
    plt.title('Error Function')
    plt.xlabel('Samples')
    plt.ylabel('MSE')
    
    # plot Coefficients
    plt.subplot(212)
    nTaps = W.shape[0]
    for row in range(0, nTaps):
        plt.plot(W[row,0:plotLen], linewidth=1)
    lgnd = [ 'w = '+str(np.around(W[i,-1],3)) for i in range(0,nTaps) ]
    plt.legend(lgnd, loc='right')
    plt.xlim(0,plotLen)
    plt.grid(True)
    plt.title('Filter Weights')
    plt.xlabel('Samples')
    plt.ylabel('Estimated Filter Weights')
    plt.tight_layout()
    