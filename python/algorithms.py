# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 22:17:21 2018

@author: Maximilian Weber

Adaptive Filter, SoSe 2018, Prof. Sikora
File: Algorithmen
"""

import numpy as np

#####
def lmsAlg(N, mu, X, D, w_init):
    """
    LMS Algorithm
    """
    
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
        y = np.dot(x.T, w)
        
        # Fehler
        e = D[:,i] - y
        
        # Adaption der Koeffizienten
        w = w[0] + mu * e * x
        
        # Chronologisches Speichern
        W[:,i] = w 
        E[i] = e
        Yd[i] = y
    
    return(E,W,w,Yd)