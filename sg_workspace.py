# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 22:14:53 2021

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
#%% import and plot signal
# signal = dfAvg[3]
plt.figure(1) 
plt.plot(signal)
#%%
nOpt = sgfilter.n_opt(signal, 2)
filterOpt = sgfilter.sg_filter_gram(signal, nOpt, 2)

plt.figure(10)
plt.plot(signal, label = 'signal with noise')
plt.plot(filterOpt, label = 'filter')
#%% debug weight function to integrate algebraic definition of Gram Polys
n = 2
i = 1
t = 1
m = 100
k = np.arange(0, n)
weightSum = 0
for k in range(0, n+1):
    print(k)
    coeff1 = (2*k+1)*sgfilter.gen_fact(2*m, k)/sgfilter.gen_fact(2*m+k+1, k+1)
    print(coeff1)
    weightSum = weightSum + (coeff1 * 
                             sgfilter.gram_poly(i,m,k,0, method = 'new')*
                             sgfilter.gram_poly(t,m,k,0, method = 'new'))
#%% handle possibilities where b>a
a = np.arange(100, 0, -1)
b = np.arange(0, 100)
diff = np.clip(a-b, 0, None)
gf = factorial(a)/factorial(diff)
bo = diff>0
gfClip = gf[diff > 0]
#%% vectorize the gram poly function with respect to i and t
k = 2
m = 100
i = np.arange(-m, m+1)
mAndI = m + i
if k == 0:
    gram_poly = 1
else:
    jArr = np.arange(0, k + 1)
    terms = np.zeros((k + 1, len(i)))
    for idx, j in enumerate(jArr):
        jFact = factorial(j)
        gf1 = sgfilter.gen_fact(j + k, 2*j)
        print(m + i, j)
        gf2 = sgfilter.gen_fact(m + i, j)
        gf3 = sgfilter.gen_fact(2*m, j)
        gfProd = gf1*gf2/gf3
        terms[idx] = ((-1)**(j + k))/(jFact**2)*gfProd
        # print(termTest)
    gram_poly = np.sum(terms, axis = 0)
    # print(gram_poly)
#%% gen fact reproduce above error
test = sgfilter.gen_fact(i + m, 4)
#%% make sure gram poly is returning proper values compared to old function
vector = sgfilter.gram_poly(i, 100, 3, 0, method = 'vector')
single = sgfilter.gram_poly(10, 100, 3, 0, method = 'new')
#%% test out then new vectorization with calculating the weight function
m = 100
i = np.arange(0, 2*m+1)
t = np.arange(0, 2*m+1)
k = np.arange(0, n)
# weightSum = 0
for k in range(0, n+1):
    print(k)
    coeff1 = (2*k+1)*sgfilter.gen_fact(2*m, k)/sgfilter.gen_fact(2*m+k+1, k+1)
    # print(i, m)
    weightSum = weightSum + (coeff1 * 
                             sgfilter.gram_poly(i,m,k,0, method = 'vector')*
                             sgfilter.gram_poly(np.array([-m]),m,k,0, method = 'vector'))
#%% now make sure the table is constructed properly
weightArr = np.zeros((2*m+1, 2*m+1))
for i in range(-m, m+1):
    for t in range(-m, m+1):
        weightArr[i + m, t + m] = 1*sgfilter.conv_weight(i,t,m,n,0)
#%% shift weight array indexing to avoid negatives
weightArrShift = np.zeros((2*m+1, 2*m+1))
for i in range(0, 2*m+1):
    for t in range(0, 2*m+1):
        print(t)
        weightArrShift[i, t] = 1*sgfilter.conv_weight(i-m,t-m,m,n,0)
#%% retest and verify consistency between new and old gen fact
a = 100
b = 100
gf = 1
for idx in range(a-b, a):
    print(idx+1)
    gf = gf*(idx+1)
#%% new vectorized gen fact
import scipy as sp
a = np.arange(100, 1)
b = np.arange(1, 100)
spFact = sp.math.factorial
fact = sp.vectorize(spFact, otypes='O')
gf = np.clip(fact(a)/fact(np.abs(a - b))*np.sign(a - b), 0, None)