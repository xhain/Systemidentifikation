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
def safe_div(x, y):
    """
    Avoid Divide by Zero
    
    """
    if y == 0:
        return 0
    return x / y


#####
def printProgressBar (iteration, total ,prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    @author: Greenstick (via Stackoverflow)
    via: https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    if iteration == total:
        print()


#####
def eigSpread(X,N):
    """
    Calculate Eigenvalue Spread (Moschytz, p.73, 2.4.1)
    
    """
    # Autocorrelation matrix
    R = np.outer(X[0,:N],X[0,:N])
    # Eigenvalue decomposition, w = eig.vals, v = eig.vecs
    w, v = np.linalg.eig(R)
    # Divide biggest eigenvalue by smallest eigenvalue
    return np.abs(np.amax(w) / np.amin(w))


#####
def linSmooth(x, N):
    """
    Smooth data with moving average over N values
    @author: Alleo (Stackoverflow)
    via: https://stackoverflow.com/a/27681394/4888669
    
    """
    csum = np.cumsum(np.insert(x, 0, 0)) 
    return (csum[N:] - csum[:-N]) / float(N)


#####
def replaceZeroes(x):
    """
    Replaces zeros from data with smallest non-zero value from input
    
    """
    min_nonzero = np.min(x[np.nonzero(x)])
    x[x == 0] = min_nonzero
    return x


####
def SNRdB(X,N):
    """
    Calculate Signal-to-Noise-Ratio
    Returns in dB
    
    """
    # Power of Signal and Noise
    Xp = np.square(X).mean()
    Np = np.square(N).mean()
    
    if Np == 0:
        Np = np.finfo(float).tiny
    
    # Calculate SNR
    SNRxn = 10 * np.log10(Xp / Np)

    return np.round(SNRxn,2)


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
    
    SNR = SNRdB(x,noise)
    
    # Add noise to input signal
    x_noise = x + noise
    return x_noise, SNR


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
def plot(X,title='No Title',style='lin',xLim=400, xlab='Samples', ylab='unspecified'):
    """
    Plot Vectors from Array
    
    """
    fig = plt.figure(figsize=(10, 2))
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
    fig.tight_layout()

    
#####
def plotdb(X,title='No Title', xLim=400):
    """
    Plot dB scale from vector X
    
    """
    X = X / np.amax(np.abs(X))
    X = 20 * np.log10(np.abs(X))
    fig = plt.figure(figsize=(12, 3))
    plt.title(title)
    plt.plot(X[0,:], linewidth=1)
    fig.tight_layout()

    
    
#####
def hist(X, title):
    """
    Quick Histrogram
    
    """
    plt.figure(figsize=(10, 2))
    plt.title(title)
    plt.hist(X[0], facecolor='b')
    plt.grid(True)
    
    
#####
def errorPlot(E, W, plotLen=500, title='No Title Set',style='lin',avgFrom=2000,avgTo=None):
    """
    Plots Learning Curve & Weights over Iterations
    
    """
    fig = plt.figure(figsize=(10, 6))
    fig.suptitle(title, fontsize=14)
    plt.subplot(211)
    
    # Preparation (trim N zeros from start)
    Ez = np.trim_zeros(E, 'f')
    
    # Moving Average Smoothing of Plot (Moschytz, p.153/154)
    Ez = linSmooth(Ez, 30)
    
    # average to?
    if avgTo == None:
        avgTo = plotLen
    
    # linear or log scale?
    if style == 'log':
        MSEunit = ' dB'
        Ez = replaceZeroes(Ez)
        #En = np.divide(Ez,Ez[0])
        # convert do decibel scale
        Eplot = 10 * np.log10(Ez/Ez[0])
        plt.ylabel('MSE (dB)')
        
    elif style == 'lin':
        MSEunit = ''
        plt.ylabel('MSE')
        Eplot = Ez
    
    EavgOpt = np.average(Eplot[avgFrom:])
    EavgAll = np.average(Eplot[:avgTo])
    # Plot Error
    plt.plot(Eplot[0:plotLen], 'b', linewidth=1)
    # Plot Average Error
    plt.plot([avgFrom, plotLen], [EavgOpt, EavgOpt], 'r--', linewidth=1.2)
    plt.plot([0, avgTo], [EavgAll, EavgAll], 'g--', linewidth=1.2)
    plt.xlim(0,plotLen)
    #plt.ylim(-100, 0)
    plt.grid(True)
    plt.title('Learning Curve')
    plt.xlabel('Samples')
    avgOptStr = 'avgOpt(MSE) = '+str(EavgOpt.round(3))+str(MSEunit)
    avgAllStr = 'avgAll(MSE) = '+str(EavgAll.round(3))+str(MSEunit)
    lgnd = ['MSE', avgOptStr, avgAllStr]
    plt.legend(lgnd, loc='right', bbox_to_anchor=(1, 1.2))
    
    # plot Weights
    Wplot = []
    plt.subplot(212)
    nTaps = W.shape[0]
    for row in range(0, nTaps):
        Wplot = np.trim_zeros(W[row,:], 'f')
        Wplot = np.insert(Wplot, 0, 0., axis=0)
        plt.plot(Wplot, linewidth=1)
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

print('*** Toolbox succesfully loaded.')