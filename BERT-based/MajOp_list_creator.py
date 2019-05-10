#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 04:15:15 2019

@author: Ani
"""

import pickle
import os

with open('/Users/Ani/Desktop/metadata/bb2genis.pkl', 'rb') as f:
    data = pickle.load(f)
    f.close()


path = '/Users/Ani/Desktop/circuit-cases/sentences_new/'


def create_MajOp_list(path):
    
    lis = []
    
    for i in range(1891,2014):
        
        folder = 'sent_'+str(i)+'/'
        a = os.listdir(path+folder)
        
        for b in a:
            if 'MajOp' in b and b[:b.index('_')] in data:
                lis.append([path+folder+b,b[:b.index('_')],data[b[:b.index('_')]]])
                
    return lis

lis = create_MajOp_list(path)

file = open('/Users/Ani/Desktop/MajOp_lis.pkl','wb')
pickle.dump(lis,file)
file.close()
        
        
        