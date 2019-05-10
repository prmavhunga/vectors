#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 10:24:26 2019

@author: Ani
"""
import pickle
import matplotlib.pyplot as plt
import torch
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

with open('/Users/Ani/Desktop/bert-topic-vecs.pkl', 'rb') as f:
    v = pickle.load(f)
    f.close()
    
vecs = [[],[],[],[],[],[],[],[],[],[]]
    
for i in range(10):
    if i!=0 and i!=8:
        for a in v[i]:
            vecs[i].append([a[0],a[1].numpy()])

del vecs[8]
del vecs[0]

matrix = np.zeros((8,30,3072))

for i in range(8):
    for j in range(30):
        matrix[i][j] = vecs[i][j][1]
        
Y_1 = PCA(n_components=20).fit_transform(matrix[0])
Y_2 = PCA(n_components=20).fit_transform(matrix[1])
Y_3 = PCA(n_components=20).fit_transform(matrix[2])
Y_4 = PCA(n_components=20).fit_transform(matrix[3])
Y_5 = PCA(n_components=20).fit_transform(matrix[4])
Y_6 = PCA(n_components=20).fit_transform(matrix[5])
Y_7 = PCA(n_components=20).fit_transform(matrix[6])
Y_8 = PCA(n_components=20).fit_transform(matrix[7])        
        
        

X_1 = TSNE(n_components=2).fit_transform(Y_1)
X_2 = TSNE(n_components=2).fit_transform(Y_2)
X_3 = TSNE(n_components=2).fit_transform(Y_3)
X_4 = TSNE(n_components=2).fit_transform(Y_4)
X_5 = TSNE(n_components=2).fit_transform(Y_5)
X_6 = TSNE(n_components=2).fit_transform(Y_6)
X_7 = TSNE(n_components=2).fit_transform(Y_7)
X_8 = TSNE(n_components=2).fit_transform(Y_8)

plt.plot(X_1.T[0],X_1.T[1],'r+', X_2.T[0],X_2.T[1],'b+',
         X_3.T[0],X_3.T[1],'g+',X_4.T[0],X_4.T[1],'y+',
         X_5.T[0],X_5.T[1],'c+', X_6.T[0],X_6.T[1],'m+',
         X_7.T[0],X_7.T[1],'k+',X_8.T[0],X_8.T[1],'m*')

                
                