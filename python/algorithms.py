# -*- coding: utf-8 -*-
"""
Created on Mo Jul 31 22:17:21 2018

@author: Maximilian Weber

Adaptive Filter, SoSe 2018, Prof. Sikora
File: Algorithmen (LMS, RLS)
"""

import numpy as np


#####
def klmsAlg(N, mu, X, D, w_init, kType='Gaussian'):
    """
    KLMS Algorithm
    Nach Haykin, Liu, Ch.2.7, p.48
    """ 
    
    # Memo: http://crsouza.com/2010/03/17/kernel-functions-for-machine-learning-applications/#kernel_functions
    
    w = w_init
    Xlen = X.shape[1]
    W = np.zeros((N, Xlen))
    E = np.zeros((Xlen, 1))
    
    if kType == 'Gaussian':
        
    elif kType == 'Polynomial':
        
    elif kType == 'Laplacian':
        
    elif kType == 'Multiquadratic'
    


#####
def rlsAlg(N, mu, X, D, w_init): 
    """
    RLS Algorithm
    Nach Moschytz Ch.4.2, p.137
    """ 
    # Initialize values
    w = w_init 
    Xlen = X.shape[1]
    eta = 100000
    R_inv = eta * np.eye(N)
    W = np.zeros((N, Xlen))
    E = np.zeros((Xlen, 1))

    # Update Loop RLS
    for i in range(N,Xlen):
        
        # Eingangsvektor
        x = X[:,i-N:i][0]
        x = x[::-1]
        
        # A priori Output value
        y = np.dot(x,w)
        
        # A priori error
        e = D[:,i-1] - y
        
        # filtered normalized data vector
        z = np.dot(R_inv, x) / (1 + np.dot(x, R_inv).dot(x))
        
        # Adjust weight
        w = w + e * z
        
        # Adjust inverse of autocorrelation
        R_inv = R_inv - z * R_inv * x
        
        # Save MSE and weight for return
        E[i] = np.square(e)
        W[:,i] = w
    print('* RLS: N = '+str(N)+', mu = '+str(mu)+', w = '+str(w))
    return(E, W, w, R_inv)


#####
def lmsAlg(N, mu, X, D, w_init):
    """
    LMS Algorithm
    Nach Moschytz Ch.3.1, p.85
    """
    # Initialize values
    w = w_init
    Xlen = X.shape[1]
    W = np.zeros((N, Xlen))
    E = np.zeros((Xlen, 1))
    Yd = np.zeros((Xlen, 1))
    
    # Update Loop LMS
    for i in range(N,Xlen):
        
        # Input vector
        x = X[:,i-N:i][0]
        x = x[::-1]
        
        # Output value
        y = np.dot(x,w)
        
        # Calculate error
        e = D[:,i-1] - y
        
        # Adjust the weight
        w = w + mu * e * x
        
        # Save MSE and weight for return
        W[:,i] = w
        E[i] = np.square(e)
        Yd[i] = y
        
    print('* LMS: N = '+str(N)+', mu = '+str(mu)+', w = '+str(w))
    return(E, W, w, Yd)