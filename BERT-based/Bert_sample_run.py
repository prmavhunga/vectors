#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 07:33:08 2019

@author: Ani
"""

import pickle
from flair.embeddings import BertEmbeddings
from flair.data import Sentence

for i in range(10):        
    
    if i!=0 and i!=8:
        file = open('/Users/Ani/Desktop/Geniss/geniss'+str(i)+'.pkl','wb')
        geniss = pickle.load(geniss[i],file)
        file.close()