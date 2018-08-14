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
def linSmooth(x, N):
    """
    Smooth data with moving average over N values
    
    """
    csum = np.cumsum(np.insert(x, 0, 0)) 
    return (csum[N:] - csum[:-N]) / float(N)

#####
def replaceZeroes(x):
    """
    Replace Zeros from data
    
    """
    min_nonzero = np.min(x[np.nonzero(x)])
    x[x == 0] = min_nonzero
    return x


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
    """
    Quick Histrogram
    
    """
    plt.figure(figsize=(12, 4))
    plt.title(title)
    plt.hist(X[0], facecolor='b')
    plt.grid(True)
    
#####
def errorPlot(E, W, plotLen=500, title='No Title Set',style='lin'):
    """
    Plots Learning Curve & Weights over Iterations
    
    """
    fig = plt.figure(figsize=(12, 6))
    fig.suptitle(title, fontsize=14)
    plt.subplot(211)
    
    # Preparation (trim N zeros from start)
    Ez = np.trim_zeros(E, 'f')
    # Moving Average Smoothing of Plot (Moschytz, p.153/154)
    Ez = linSmooth(Ez, 30)
    
    # linear or log scale?
    if style == 'log':
        # Normalize to Maxmimum Error
        maxE = np.amax(Ez)
        En = np.true_divide(Ez,maxE)
        # convert do decibel scale
        En = replaceZeroes(En)
        Eplot = 20 * np.log10(En)
        plt.ylabel('MSE (dB)')
        
    elif style == 'lin':
        plt.ylabel('MSE')
        Eplot = Ez
    
    # Plot Error
    plt.plot(Eplot[0:plotLen], 'b', linewidth=1)
    plt.xlim(0,plotLen)
    #plt.ylim(-100, 0)
    plt.grid(True)
    plt.title('Learning Curve')
    plt.xlabel('Samples')
    plt.legend(['avg(E) = '+str(np.average(Eplot).round(2))], loc='right', bbox_to_anchor=(1, 1.1))
    
    
    # plot Weights
    plt.subplot(212)
    nTaps = W.shape[0]
    for row in range(0, nTaps):
        plt.plot(W[row,0:plotLen], linewidth=1)
    lgnd = [ 'w = '+str(np.around(W[i,-1],3)) for i in range(0,nTaps) ]
    plt.legend(lgnd, loc='right',title="Final Weights")
    plt.xlim(0,plotLen)
    plt.grid(True)
    plt.title('Filter Weights')
    plt.xlabel('Samples')
    plt.ylabel('Estimated Weight')
    
    # Layouting
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    