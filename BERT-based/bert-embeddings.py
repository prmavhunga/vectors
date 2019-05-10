#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 11:10:14 2019

@author: Ani
"""

from flair.embeddings import BertEmbeddings
from flair.data import Sentence
import torch

#example path. Can be any text file
path = '/Users/Ani/Desktop/circuit-cases/sentences_new/sent_1999/X3AV14_contentMajOp_SELYA.txt'


# initialize embedding
embedding = BertEmbeddings('bert-base-uncased')
#To avoid overflow. OBSERVED OVERFLOW AT >=300. Better if less than 250.
MAX = 200

#for below function:
# output: embed success2 - (No problems)
# output: embed success1 - (number of sentences smaller than 3 in text. No problems)
# output: bad sentences - (there is atleast one sentence that expands to a sentence longer than MAX
#   after tokenisation for BERT. This case returns null torch tensor - torch.tensor([]))

def bert_doc_embed(path):
    
    f = open(path,'r')
    f1 = f.readlines()
    f.close() 

    #number of sentences in text file
    l = len(f1)
    print('number of sentences in text file: '+str(l))
    
    diff = l%3
    quo = int((l-diff)/3)
    
    #number of tokens
    token_count = 0
    
    if quo == 0:
        f2 = f1[0]
        for k in range(diff-1):
            f2 = f2 + f1[k+1]
            
        #create sentence
        sentence = Sentence(f2)
        
        size = len(sentence)
        if size<MAX:
            token_count = size
        
            #embed words in sentence
            embedding.embed(sentence)
        
            A = sentence[0].embedding
            for j in range(size-1):
                A = A + sentence[j+1].embedding
        else:
            sentence = Sentence(f1[0])
            size = len(sentence)
            if size>MAX:
                print('bad sentences')
                return torch.tensor([])
            token_count = token_count + size
                
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
        sentence = Sentence(f1[0]+f1[1]+f1[2])
        
        size = len(sentence)
        if size<MAX:
            token_count = token_count + size
        
        
            #embed words in sentence
            embedding.embed(sentence)
        
            A = sentence[0].embedding
            for j in range(size-1):
                A = A + sentence[j+1].embedding
        else:
            sentence = Sentence(f1[0])
            size = len(sentence)
            if size>MAX:
                print('bad sentences')
                return torch.tensor([])
            token_count = token_count + size
                
            #embed words in sentence
            embedding.embed(sentence) 
            
            A = sentence[0].embedding
            for j in range(size-1):
                A = A + sentence[j+1].embedding
            
            sentence = Sentence(f1[1])
            size = len(sentence)
            if size>MAX:
                print('bad sentences')
                return torch.tensor([])
             
            token_count = token_count + size
                
            #embed words in sentence
            embedding.embed(sentence) 
            
            for j in range(size):
                A = A + sentence[j].embedding 
                
            sentence = Sentence(f1[2])
            size = len(sentence)
            if size>MAX:
                print('bad sentences')
                return torch.tensor([])
             
            token_count = token_count + size
                
            #embed words in sentence
            embedding.embed(sentence) 
            
            for j in range(size):
                A = A + sentence[j].embedding             
            
        for i in range(quo-1):
        
            #create a sentence
            sentence = Sentence(f1[3*(i+1)]+f1[3*(i+1)+1]+f1[3*(i+1)+2])
        
            size = len(sentence)
            if size<MAX:
                token_count = token_count + size
        
                #embed words in sentence
                embedding.embed(sentence)
            
                for j in range(size):
                    A = A + sentence[j].embedding
            
            else:
                sentence = Sentence(f1[3*(i+1)])
                size = len(sentence)
                if size>MAX:
                    print('bad sentences')
                    return torch.tensor([])
                 
                token_count = token_count + size
                
                #embed words in sentence
                embedding.embed(sentence) 
            
                for j in range(size):
                    A = A + sentence[j].embedding
 
                sentence = Sentence(f1[3*(i+1)+1])
                size = len(sentence)
                if size>MAX:
                    print('bad sentences')
                    return torch.tensor([])
                 
                token_count = token_count + size
                
                #embed words in sentence
                embedding.embed(sentence) 
            
                for j in range(size):
                    A = A + sentence[j].embedding
                     
                sentence = Sentence(f1[3*(i+1)+2])
                size = len(sentence)
                if size>MAX:
                    print('bad sentences')
                    return torch.tensor([])
 
                token_count = token_count + size
                
                #embed words in sentence
                embedding.embed(sentence) 
            
                for j in range(size):
                    A = A + sentence[j].embedding
                      
        
        if diff!=0:           
            f2 = f1[quo*3]        
            for i in range(diff-1):
                f2 = f2 + f1[3*quo+i+1]
        
            #create sentence
            sentence = Sentence(f2)
        
            size = len(sentence)
            if size<MAX:
                
                token_count = token_count + size
        
                #embed words in sentence
                embedding.embed(sentence)
        
                for j in range(size):
                    A = A + sentence[j].embedding
            
        A = A/token_count
        print('embed success2')
        return A
              
B = bert_doc_embed(path)
    