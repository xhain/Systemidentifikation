# -*- coding: utf-8 -*-
"""
Created on Mo Jul 31 22:17:21 2018

@author: Maximilian Weber

Adaptive Filter, SoSe 2018, Prof. Sikora
File: Algorithmen (LMS, RLS)
"""

import numpy as np
import kernels as ks

#####

# via https://github.com/pin3da/kernel-adaptive-filtering/blob/master/filters.py
class klmsAlgo(ks.Kernel):
    def __init__(
        self,
        N,
        first_input=None,
        first_output=None,
        mu=0.5,
        sigma=1
    ):
        if first_input is not None:
            self.inputs = [first_input]
        else:
            self.inputs = [np.zeros(N)]
        if first_output is not None:
            self.weights = [first_output * mu]
        else:
            self.weights = [0]
        self.mu = mu
        self.sigma = sigma
        self.error = None

    def predict(self, new_input):
        estimate = 0
        for i in range(0, len(self.weights)):
            addition = self.weights[i] * self.kernel(self.inputs[i], new_input)
            estimate += addition
        return estimate

    def update(self, new_input, expected):
        self.error = expected - self.predict(new_input)
        self.inputs.append(new_input)
        new_weights = self.mu * self.error
        self.weights.append(new_weights)

    def name(self):
        return 'KLMS'
    


#####
def rlsAlg(N, X, D, w_init): 
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
        
    print('* RLS: N = '+str(N)+', w = '+str(w))
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