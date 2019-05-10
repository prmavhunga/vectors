#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 07:11:27 2019

@author: Ani
"""

import pickle
import math

with open('/Users/Ani/Desktop/MajOp_lis.pkl', 'rb') as f:
    lis = pickle.load(f)
    f.close()

geniss = [[],[],[],[],[],[],[],[],[],[]]

count = 0;

for a in lis:
    if math.isnan(a[2]):
        count = count+1      
    else: 
        geniss[int(a[2])].append(a[0])
        

#for i in range(10):        
#    
#    if i!=0 and i!=8:
#        file = open('/Users/Ani/Desktop/Geniss/geniss'+str(i)+'.pkl','wb')
#        pickle.dump(geniss[i],file)
#        file.close()
        
file = open('/Users/Ani/Desktop/Geniss/geniss.pkl','wb')
pickle.dump(geniss,file)
file.close()