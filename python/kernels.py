#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 23:53:35 2018

@author: max
File: Kernels
"""

import numpy as np


# Memo: http://crsouza.com/2010/03/17/kernel-functions-for-machine-learning-applications/#kernel_functions

#####
class Kernel:
    def gaussK(self, a, b):
        norm = np.linalg.norm(a - b)
        term = np.square(norm) / (2 * np.square(self.sigma))
        return np.exp(-1 * term)
    
    def laplaceK(self, a, b):
        norm = np.linalg.norm(a - b)
        term = np.square(norm) / (self.sigma)
        return np.exp(-1 * term)

