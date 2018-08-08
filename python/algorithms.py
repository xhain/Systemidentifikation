# -*- coding: utf-8 -*-
"""
Created on Mo Jul 31 22:17:21 2018

@author: Maximilian Weber

Adaptive Filter, SoSe 2018, Prof. Sikora
File: Algorithmen (LMS, RLS)
"""

import numpy as np


####
def rlsAlg(N, mu, X, D, w_init): 
    """
    RLS Algorithm
    Nach Moschytz 4.2, p.137
    """
    
    print('*** Starting RLS adaption...')
    
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
        
    print('*** ...RLS done.')
    return(E, W, w, R_inv)


#####
def lmsAlg(N, mu, X, D, w_init):
    """
    LMS Algorithm
    Nach Moschytz 3.1, p.85
    """
    
    print('*** Starting LMS adaption...')
    
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
        
    print('*** ...LMS done.')
    return(E, W, w, Yd)