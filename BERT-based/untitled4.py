#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 18:33:58 2019

@author: Ani
"""

import pickle
import matplotlib.pyplot as plt
import torch
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

with open('/Users/Ani/Desktop/bert-topic-vecs2.pkl', 'rb') as f:
    v = pickle.load(f)
    f.close()
    
vecs = [[],[],[],[],[],[],[],[],[],[]]
    
for i in range(10):
    if i!=0 and i!=8:
        for a in v[i]:
            vecs[i].append([a[0],a[1].numpy()])

del vecs[8]
del vecs[0]

start = 1891
year_avg = np.zeros((123,3072))
year_ct = np.zeros(123)

for i in range(8):
    for j in range(100):
        if np.linalg.norm(vecs[i][j][1]) != 0:
            vecs[i][j][1] = (vecs[i][j][1]/np.linalg.norm(vecs[i][j][1]))

for i in range(8):
    for a in vecs[i]:
        year = int(a[0][52:56])
        year_avg[year-start] = year_avg[year-start] + a[1]
        year_ct[year-start] = year_ct[year-start]+1
        
for i in range(123):
    if year_ct[i] == 0:
        year_ct[i] = 1
        
    year_avg[i] = year_avg[i]/year_ct[i]
    
vecs2 = vecs

for i in range(8):
    for j in range(100):
       year = int(vecs2[i][j][0][52:56])
       vecs2[i][j][1] = vecs2[i][j][1] - year_avg[year-start]
       
file = open('/Users/Ani/Desktop/bert-topic-vecs-year-centered.pkl','wb')
pickle.dump(vecs2,file)
file.close()  
       



