#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 22:10:21 2018

@author: Maximilian Weber
@
"""

# Import tools
import tools as ts
import algorithms as algo

import numpy as np


# IMPORT DATA
filepath = './data/'
importMat, fileNames = ts.importmat(filepath)

# Load Input
x_training = importMat['Training']['x_training']
x_test = importMat['Test']['x_test']

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

# Remember: System Change doesn't have to be manually induced. H_*IR2_D already changes from one to the other.
# Proof: plotvecs([H_FIR1_D.T - H_FIR2_D.T]), plotvecs([H_IIR1_D.T - H_IIR2_D.T])


# FIR LMS
w_init = np.array([[1, 1, 1, 1, 1]])
mu = 0.01
E, W, w, Yd = algo.lmsAlg(5, mu, H_FIR1_X, H_FIR1_D, w_init)
ts.plotvecs(E.T,'FIR Konstant')


# FIR LMS Systemwechsel
w_init = np.array([[1, 1, 1, 1, 1]])
E, W, w, Yd = algo.lmsAlg(5, mu, H_FIR2_X, H_FIR2_D, w_init)
ts.plotvecs(E.T,'FIR Systemwechsel')


# IIR LMS 
w_init = np.array([[1, 1, 1, 1, 1]])
E, W, w, Yd = algo.lmsAlg(5, mu, H_IIR1_X, H_IIR1_D, w_init)
ts.plotvecs(E.T,'IIR Konstant')


# IIR LMS Systemwechsel
w_init = np.array([[1, 1, 1, 1, 1]])
E, W, w, Yd = algo.lmsAlg(5, mu, H_IIR2_X, H_IIR2_D, w_init)
ts.plotvecs(E.T,'IIR Systemwechsel')