#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 11:10:14 2019

@author: Ani
"""

from flair.embeddings import BertEmbeddings
from flair.data import Sentence


# initialize embedding
embedding = BertEmbeddings('bert-base-uncased')

#path = '/Users/Ani/Desktop/circuit-cases/sentences_new/sent_2008/X1A1MP8003_contentMajOp_DAMON J. KEITH.txt'

def bert_doc_embed(path):
    
    f = open(path,'r')
    f1 = f.readlines()
    f.close() 

    #number of sentences in text file
    l = len(f1)
    print('number of sentences in text file: '+str(l))
    
    diff = l%5
    quo = int((l-diff)/5)
    
    #number of tokens
    token_count = 0
    
    if quo == 0:
        f2 = f1[0]
        for k in range(diff-1):
            f2 = f2 + f1[k+1]
            
        #create sentence
        sentence = Sentence(f2)
        
        size = len(sentence)
        token_count = size
        
        #embed words in sentence
        embedding.embed(sentence)
        
        A = sentence[0].embedding
        for j in range(size-1):
            A = A + sentence[j+1].embedding
                
        A = A/token_count
        
        print('embed success1')
        return A
    
    else:
        
        #create a sentence
        sentence = Sentence(f1[0]+f1[1]+f1[2]+f1[3]+f1[4])
        
        size = len(sentence)
        token_count = token_count + size
        
        #embed words in sentence
        embedding.embed(sentence)
        
        A = sentence[0].embedding
        for j in range(size-1):
            A = A + sentence[j+1].embedding
            
                
        for i in range(quo-1):
        
            #create a sentence
            sentence = Sentence(f1[5*(i+1)]+f1[5*(i+1)+1]+f1[5*(i+1)+2]+f1[5*(i+1)+3]+f1[5*(i+1)+4])
        
            size = len(sentence)
            token_count = token_count + size
        
            #embed words in sentence
            embedding.embed(sentence)
            
            for j in range(size):
                A = A + sentence[j].embedding
        
        if diff!=0:           
            f2 = f1[quo*5]        
            for i in range(diff):
                f2 = f2 + f1[5*quo+i]
        
            #create sentence
            sentence = Sentence(f2)
        
            size = len(sentence)
            token_count = token_count + size
        
            #embed words in sentence
            embedding.embed(sentence)
        
            for j in range(size):
                A = A + sentence[j].embedding
            
        A = A/token_count
        print('embed success2')
        return A
        
B = bert_doc_embed(path)
    