#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 22:10:21 2018

@author: Maximilian Weber
File: Main - running tests and simulations here next to documentation in Notebook
"""

# Import tools
import tools as ts
import algorithms as algo

import numpy as np
#import matplotlib.pyplot as plt
#from scipy import signal as sig


# IMPORT DATA
filepath = './data/'
importMat, fileNames = ts.importmat(filepath)
    
# Load Input
x_training = importMat['Training']['x_training'].T
x_test = importMat['Test']['x_test'].T

# Load FIR Systems
H_FIR1_D = importMat['System_FIR27']['D_']
H_FIR1_X = importMat['System_FIR27']['X']

# Vgl Gabriel
H_FIRg_D = importMat['System_FIR3']['D_']
H_FIRg_X = importMat['System_FIR3']['X']

H_FIR2_D = importMat['Systemwechsel_FIR27']['D_']
H_FIR2_X = importMat['Systemwechsel_FIR27']['X']

# Load IIR Systems
H_IIR1_D = importMat['System_IIR27']['D_']
H_IIR1_X = importMat['System_IIR27']['X']

H_IIR2_D = importMat['Systemwechsel_IIR27']['D_']
H_IIR2_X = importMat['Systemwechsel_IIR27']['X']


# Add noise to receiver signal
variance = 0.1
H_FIR1_Dn, SNR1 = ts.addNoise(H_FIR1_D,variance)


# Initialize Test
N = 5
w_init = np.zeros(N)
mu = 0.01
plotLen = 500


## FIR LMS
E, W, w, Yd = algo.lmsAlg(N, mu, H_FIR1_X, H_FIR1_Dn, w_init)
ts.errorPlot(E, W, plotLen,'LMS Lernkurve für FIR-System, N = '+str(N), style='lin')

#print('Kond: ', ts.eigSpread(H_FIR1_D,1000) )
E, W, w, Yd = algo.rlsAlg(N, H_FIR1_X, H_FIR1_Dn, w_init)
ts.errorPlot(E, W, plotLen,'RLS Lernkurve für FIR-System, N = '+str(N), style='lin')



# KLMS
#ts.plot(x_training.T,'training')
#ts.plot(x_test.T,'test')

Kern = algo.klmsAlgo(N,'gauss', mu=0.01, sigma=3)
Kern, Et, Wt, Yt = algo.Klearn(Kern, N, x_training, x_training)

Kern, Ep, Wp, Yp = algo.Kpredict(Kern, N, x_test, x_test)
ts.plot(Ep.T, 'KLMS Lernkurve für N = '+str(N),xLim=10000)

E, W, w, Yd = algo.lmsAlg(N, mu, x_test, x_test, w_init)
ts.errorPlot(E, W, 10000,'LMS Lernkurve für KLMS Vergleich, N = '+str(N), style='lin')






