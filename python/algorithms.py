# -*- coding: utf-8 -*-
"""
Created on Mo Jul 31 22:17:21 2018

@author: Maximilian Weber

Adaptive Filter, SoSe 2018, Prof. Sikora
File: Algorithmen (LMS, RLS)
"""

import numpy as np
import kernels as ks
import tools as ts


#####
def Kpredict(Kern, N, X):
    """
    KLMS online test/predict routine
    """ 
    Xlen = X.shape[1]
    Kern.prediction = [0]
    Kern.errors = [0] 

    for i in range(N,Xlen):
        
        # cut a data chunk
        x = X[:,i-N:i][0]
        x = x[::-1] # flip
        
        y = Kern.predict(x)
        
        # calculate error
        Kern.error = X[:,i] - y
        Kern.prediction.append(y)
        
        # save Test Error and Prediction
        Kern.errors.append(np.square(Kern.error))
        
        # Progress Bar
        ts.printProgressBar(i,Xlen,prefix='KLMS Predicting',length=25)


#####
def Klearn(Kern, N, X):
    """
    KLMS online update/train routine
    """ 
    Xlen = X.shape[1]
    
    for i in range(N,Xlen):
        
        # cut a data chunk
        x = X[:,i-N:i][0]
        x = x[::-1] # flip
        
        # update for chunk
        Kern.update(x, X[:,i])
        
        # Progress Bar
        ts.printProgressBar(i,Xlen,prefix='KLMS Learning',length=25)


#####
class klms(ks.Kernel):
    """
    KLMS class
    Nach Haykin, Liu, Principe, p.34 / Algorithm 2
    """
    def __init__(
        self,
        N = 5,
        kFun = 'gauss',
        mu = 0.5,
        sigma = 1
    ):
        self.data = [0]
        self.weights = [0]
        self.mu = mu
        self.sigma = sigma
        self.error = None
        self.prediction = [0] 
        self.errors = [0]
        
        # choose kernel type (to be expanded)
        if kFun == 'gauss':
            self.kFun = self.gaussK
        elif kFun == 'laplace':
            self.kFun = self.laplaceK
        else:
            self.kFun = self.gaussK
    
    def predict(self, x):
        
        # initialize estimate
        predict = 0
        for i in range(len(self.weights)):
            # predict for every datapoint i with according weight (past)
            predict_i = self.weights[i] * self.kFun(self.data[i],x)
            # sum all datapoints running i to estimate prediction
            predict += predict_i
        return predict
    
    def update(self, x, desired):
        # calculate error for current prediction towards desired output
        self.error = desired - self.predict(x)
        # calculate weight based on error & learning rate
        new_weight = self.mu * self.error
        # add weights to stack (past)
        self.weights.append(new_weight)
        # add datapoint to stack (past)
        self.data.append(x)
        # add prediction to stack
        self.prediction.append(self.predict(x))
        self.errors.append(self.error**2)


#####
def rlsAlg(N, X, D, w_init, memleak=0.0): 
    """
    RLS Algorithm
    Nach Moschytz Ch.4.2, p.137 / Ch.4.3, p.145
    """ 
    # Initialize values
    w = w_init 
    Xlen = X.shape[1]
    eta = 100000
    R_inv = eta * np.eye(N)
    W = np.zeros((N, Xlen))
    E = np.zeros((Xlen, 1))
    
    # 0 < rho < 1, Moschytz p.145, RlS-Algo mit Vergessensfaktor
    if memleak < 0 or memleak > 1:
        raise ValueError('This parameter must be a Positive Number between 0 and 1')
    rho = np.clip(1.0 - memleak, 0.0001, 1.0)

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
        z = np.dot(R_inv, x) / (rho + np.dot(x, R_inv).dot(x))
        
        # Adjust weight
        w = w + np.multiply(e, z)
        
        # Adjust inverse of autocorrelation
        R_inv = 1/rho * (R_inv - z * R_inv * x)
        
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
    
print('*** Algorithms succesfully loaded.')