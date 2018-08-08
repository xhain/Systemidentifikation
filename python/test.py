# -*- coding: utf-8 -*-
"""
Created on Mo Aug 8 11:52:13 2018

@author: Maximilian Weber

Adaptive Filter, SoSe 2018, Prof. Sikora
File: Tests
"""

import numpy as np
import scipy.signal as sig

Hfir = np.array([0.7, 0.1, -0.03, 0.18, -0.24])
Hirr = 1 / np.array([0.82, -0.03])

Dfir = sig.lfilter(Hfir, Xfir)