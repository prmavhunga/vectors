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



                
                