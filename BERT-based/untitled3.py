#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 10:24:26 2019

@author: Ani
"""
import pickle
import matplotlib.pyplot as plt
import torch


with open('/Users/Ani/Desktop/bert-topic-vecs.pkl', 'rb') as f:
    v = pickle.load(f)
    f.close()
    
plt.plot(v[2][3][1])