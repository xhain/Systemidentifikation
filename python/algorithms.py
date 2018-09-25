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
def Kpredict(Kern, N, X, D):
    """
    KLMS online test/predict routine
    """ 
    Xlen = X.shape[1]
    E = np.zeros((Xlen, 1))
    Y = np.zeros((Xlen, 1))

    for i in range(N,Xlen):
        
        # cut a data chunk
        x = X[:,i-N:i][0]
        #x = x[::-1] # flip?
        d = D[:,i-1]
        
        # predict for chunk (legacy code)
        y = Kern.predict(x)
        Kern.error = d - y
        
        # save Test Error and Prediction
        E[i] = np.square(Kern.error)
        Y[i] = y #- X[:,i][0]
        
        ts.printProgressBar(i,Xlen,prefix='KLMS Predicting',length=25)

    return E, Y


#####
def Klearn(Kern, N, X, D):
    """
    KLMS online update/train routine
    """ 
    Xlen = X.shape[1]
    E = np.zeros((Xlen, 1))

    for i in range(N,Xlen):
        
        # cut a data chunk
        x = X[:,i-N:i][0]
        #x = x[::-1] # flip?
        
        # get desired (i.e. X, but we can add noise later if arg is seperate)
        d = D[:,i]
        
        # update for chunk
        Kern.update(x,d)

        # save Training Error
        E[i] = np.square(Kern.error)
        ts.printProgressBar(i,Xlen,prefix='KLMS Learning',length=25)
        
    return E


#####
class klms(ks.Kernel):    
    """
    KLMS class
    Nach Haykin, Liu, Principe, p.34 / Algorithm 2
    """
    
    def __init__(
        self,
        N,
        kFun = 'gauss',
        X = None,
        W = None,
        mu = 0.5, # learning rate / step size
        sigma = 1, # bandwidth
    ):
        # Prepare I/O and weights
        if X is not None:
            self.data = [X]
        else:
            self.data = np.zeros(N)
        if W is not None:
            self.weights = [W * mu]
        else:
            self.weights = np.zeros(N)
            
        self.mu = mu
        self.sigma = sigma
        self.error = None
        
        # Choose Kernel
        if kFun == 'gauss':
            self.kFun = self.gaussK
        elif kFun == 'laplace':
            self.kFun = self.laplaceK
        else:
            self.kFun = self.gaussK

    # Prediction routine
    def predict(self, new_x):
        prediction = 0
        for i in range(0, len(self.weights)):
            aufdat = self.weights[i] * self.kFun(self.data[i], new_x)
            prediction += aufdat
        return prediction

    # Update / Training routine
    def update(self, new_x, D):
        self.error = D - self.predict(new_x)
        self.data = np.append(self.data, new_x)
        weights_new = self.mu * self.error
        self.weights = np.append(self.weights, weights_new)
    

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
        R_inv = ts.safe_div(1,rho) * (R_inv - z * R_inv * x)
        
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