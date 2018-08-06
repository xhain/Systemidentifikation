#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 22:10:21 2018

@author: Maximilian Weber
@
"""


import tools as ts


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


