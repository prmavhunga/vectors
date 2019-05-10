#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 07:33:08 2019

@author: Ani
"""

import pickle
from flair.embeddings import BertEmbeddings
from flair.data import Sentence
import random
import torch


# initialize embedding
embedding = BertEmbeddings('bert-base-uncased')
#To avoid overflow. OBSERVED OVERFLOW AT >=300. Better if less than 250.
MAX = 200

#for below function:
# output: embed success2 - (No problems)
# output: embed success1 - (number of sentences smaller than 3 in text. No problems)
# output: bad sentences - (there is atleast one sentence that expands to a sentence longer than MAX-
# -after tokenisation for BERT. This case returns null torch tensor - torch.tensor([]))

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
     
    
with open('/Users/Ani/Desktop/Geniss/geniss.pkl', 'rb') as f:
    geniss = pickle.load(f)
    f.close()
       
vectors = [[],[],[],[],[],[],[],[],[],[]]

        
#number of vectors of each topic created
vec_number = 100
min_sentences = 20

for i in range(10):        
    if i!=0 and i!=8:
        random.shuffle(geniss[i])
        print('topic_'+str(i)+'started...')
        j=0
        k=0
        while j<vec_number:
            g = open(geniss[i][k],'r')
            g1 = g.readlines()
            g.close() 

            if len(g1)>=min_sentences:
                print(str(j+1)+'.'+geniss[i][k])
                vectors[i].append((geniss[i][k],bert_doc_embed(geniss[i][k])))
                j=j+1
            
            k=k+1
            
        print('topic_'+str(i)+'completed')
            
file = open('/Users/Ani/Desktop/bert-topic-vecs2.pkl','wb')
pickle.dump(vectors,file)
file.close()            
            
    