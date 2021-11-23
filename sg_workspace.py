# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 22:14:53 2021

Example workspace demonstrating how to use the optimized Savitzky-Golay filter

@author: Daniel Barnak
"""
import sgfilter 
import matplotlib.pyplot as plt

import numpy as np
import math
from scipy.special import factorial
#%% construct a test signal
randArray = np.random.rand(2000)
xpos = np.arange(0, 2000)
# signal = np.exp(0.01*xpos)/np.exp(xpos[-1]*0.01) + randArray
signal = np.sin(0.01*xpos) + randArray
#%% Find optimal window length and plot the result
nOpt = sgfilter.n_opt(signal, 2)
filterOpt = sgfilter.sg_filter_gram(signal, nOpt, 2)

plt.figure(10)
plt.plot(signal, label = 'signal with noise')
plt.plot(filterOpt, label = 'filter')
