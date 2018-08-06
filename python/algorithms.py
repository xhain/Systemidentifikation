# -*- coding: utf-8 -*-
"""
Created on Mo Jul 31 22:17:21 2018

@author: Maximilian Weber

Adaptive Filter, SoSe 2018, Prof. Sikora
File: Algorithmen
"""

import numpy as np


#####
def rlsAlgo(N, mu, X, D, w_init):
    
    p0 = 1000000;            # Initialisierung von
    inv_R = p0*eye(N);       # inv_R 
    
    adaptlen = length(X)
    w = w_start         
    W = zeros(N,adaptlen)
    E = zeros(adaptlen,1) 

# RLS Update

    for i=N:adaptlen:
        W(:,i)=w;
        x = X(i:-1:i-N+1)        #Eingangsvektor (x[k],x[k-1],..,x[k-N+1])
        y = x.T*w                  # Filterausgang
        e = D(i)-y               # Fehler
        c = 1/(rho+x.T*inv_R*x)               
        inv_R =1 /rho*(inv_R-c*inv_R*x*x.T*inv_R)  # Aufdatierung von inv_R 
        w=w+inv_R*e*x                           # Aufdatierung von w 
        E(i) = e


#####
def lmsAlg(N, mu, X, D, w_init):
    """
    LMS Algorithm
    """
    print('*** Starting LMS adaption...')
    w = w_init
    Xlen = X.shape[1]
    W = np.zeros((N, Xlen))
    E = np.zeros((Xlen,1))
    Yd = np.zeros((Xlen,1))
    
    nLen = np.linspace(N,Xlen-N,Xlen,dtype=np.int16)
    for i in nLen:
        
        # Eingangsvektor
        von = int(i-N)
        bis = int(i)
        x = X[:,von:bis]
        
        # Ausgangsvektor
        y = np.dot(x,w.T)
        
        # Fehler
        e = D[:,i] - y
        
        # Adaption der Koeffizienten
        w = w + mu * e * x
        
        # Chronologisches Speichern
        W[:,i] = w 
        E[i] = np.square(e)
        Yd[i] = y
        
    print('*** ...LMS done.')
    return(E,W,w,Yd)