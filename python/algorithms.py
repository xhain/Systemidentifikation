# -*- coding: utf-8 -*-
"""
Created on Mo Jul 31 22:17:21 2018

@author: Maximilian Weber

Adaptive Filter, SoSe 2018, Prof. Sikora
File: Algorithmen
"""

import numpy as np


####
def rlsAlg(N, mu, X, D, w_init): 
    """
    RLS Algorithm
    Nach Moschytz 4.2, p.137
    """
    
    print('*** Starting RLS adaption...')
    
    # Init
    w = w_init 
    Xlen = X.shape[1]
    rho = 10000
    R_inv = rho * np.eye(N)
    W = np.zeros((N, Xlen))
    E = np.zeros((Xlen, 1))

    # Update Loop RLS
    for i in range(N,Xlen):
        
        # Eingangsvektor
        x = X[:,i-N:i][0]
        x = x[::-1]
        
        # A priori Ausgangswert
        y = np.dot(x,w.T)
        
        # A priori Fehler
        e = D[:,i-1] - y
        
        # Gefilterter normierter Datenvektor
        z = np.dot(R_inv, x) / (1 + np.dot(x, x * R_inv))
        
        # Aufdatierung des optimalen Gewichts
        w = w + e * z
        
        # Aufdatierung der Inversen der Autokorrelationsmatrix
        R_inv = R_inv - z * np.dot(R_inv,x)
        
        
        # Chronologisches Speichern
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
    
    # Init
    w = w_init
    Xlen = X.shape[1]
    W = np.zeros((N, Xlen))
    E = np.zeros((Xlen, 1))
    Yd = np.zeros((Xlen, 1))
    
    # Update Loop LMS
    for i in range(N,Xlen):
        
        # Eingangsvektor
        x = X[:,i-N:i][0]
        x = x[::-1]
        
        # Ausgangsvektor
        y = np.dot(x,w)
        
        # Fehler
        e = D[:,i-1] - y
        
        # Adaption der Koeffizienten
        w = w + mu * e * x
        
        # Chronologisches Speichern
        W[:,i] = w
        E[i] = np.square(e)
        Yd[i] = y
        
    print('*** ...LMS done.')
    return(E, W, w, Yd)