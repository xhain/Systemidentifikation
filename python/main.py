#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 22:10:21 2018

@author: Maximilian Weber
File: Main - For running tests and debugging in Spyder
"""

# Import tools
import tools as ts
import algorithms as algo

import numpy as np
import matplotlib.pyplot as plt
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

H_FIR2_D = importMat['Systemwechsel_FIR27']['D_']
H_FIR2_X = importMat['Systemwechsel_FIR27']['X']

# Load IIR Systems
H_IIR1_D = importMat['System_IIR27']['D_']
H_IIR1_X = importMat['System_IIR27']['X']

H_IIR2_D = importMat['Systemwechsel_IIR27']['D_']
H_IIR2_X = importMat['Systemwechsel_IIR27']['X']


# Add noise to receiver signal
variance = 0.0
H_FIR1_Dn, SNR1 = ts.addNoise(H_FIR1_D,variance)

H_FIR2_Dn, SNR1 = ts.addNoise(H_FIR2_D,variance)


# Initialize Test
N = 5
w_init = np.zeros(N)
mu = 0.01
plotLen = 10000


## FIR LMS
E, W, w, Yd = algo.lmsAlg(N, mu, H_FIR1_X, H_FIR1_D, w_init, predict=0)
ts.errorPlot(E, W, plotLen,'LMS Lernkurve für FIR-System, N = '+str(N), style='lin')


# Initialize Test
N = 5
w_init = w
mu = 0.01
plotLen = 10000


## FIR LMS
E, W, w, Yd = algo.lmsAlg(N, mu, H_FIR1_X, H_FIR1_D, w, predict=True)
ts.errorPlot(E, W, plotLen,'LMS Lernkurve für FIR-System, N = '+str(N), style='lin')


# KLMS
#ts.plot(x_training.T,'training')
#ts.plot(x_test.T,'test')
#
#a = 0
#b = 500
#c = a + b
#
#traindata = x_training[:,a:c]
#testdata = x_test[:,a:c]
#
## # # # # PLOT # # # # # #
#plt.clf()
#fig = plt.figure
#plt.subplot(211)
#
#Kern = algo.klms(N, 'gauss', mu=1.0, sigma=1.0) # sig = 0.5 for N = 5
#algo.Klearn(Kern, N, traindata)
#E = Kern.errors
#
#plt.plot(traindata.T[N:],'k--')
#plt.plot(Kern.prediction,'b')
#
#algo.Kpredict(Kern, N, testdata)
#plt.plot(testdata.T[N:],'k-.')
#plt.plot(Kern.prediction,'r')
#plt.xlim([N,b])
#plt.legend(['Traindata','Training','Testdata','Test'])
#
#plt.subplot(212)
#plt.plot(E,'g')
#plt.plot(Kern.errors,'r')
#plt.ylim([0, 0.2])
#plt.xlim([N,b])
#plt.legend(['MSE Train','MSE Test'])
#
#plt.show()
#
#plt.figure()
#plt.plot(Kern.weights)