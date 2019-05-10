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


# initialize embedding
embedding = BertEmbeddings('bert-base-uncased')

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
        if size<430:
            token_count = size
        
            #embed words in sentence
            embedding.embed(sentence)
        
            A = sentence[0].embedding
            for j in range(size-1):
                A = A + sentence[j+1].embedding
        else:
            sentence = Sentence(f1[0])
            size = len(sentence)
            if size>430:
                print('bad sentences')
                return []
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
        if size<430:
            token_count = token_count + size
        
        
            #embed words in sentence
            embedding.embed(sentence)
        
            A = sentence[0].embedding
            for j in range(size-1):
                A = A + sentence[j+1].embedding
        else:
            sentence = Sentence(f1[0])
            size = len(sentence)
            if size>430:
                print('bad sentences')
                return []
            token_count = token_count + size
                
            #embed words in sentence
            embedding.embed(sentence) 
            
            A = sentence[0].embedding
            for j in range(size-1):
                A = A + sentence[j+1].embedding
            
            sentence = Sentence(f1[1])
            size = len(sentence)
            if size>430:
                print('bad sentences')
                return []
             
            token_count = token_count + size
                
            #embed words in sentence
            embedding.embed(sentence) 
            
            for j in range(size):
                A = A + sentence[j].embedding 
                
            sentence = Sentence(f1[2])
            size = len(sentence)
            if size>430:
                print('bad sentences')
                return []
             
            token_count = token_count + size
                
            #embed words in sentence
            embedding.embed(sentence) 
            
            for j in range(size):
                A = A + sentence[j].embedding             
            
        for i in range(quo-1):
        
            #create a sentence
            sentence = Sentence(f1[3*(i+1)]+f1[3*(i+1)+1]+f1[3*(i+1)+2])
        
            size = len(sentence)
            if size<430:
                token_count = token_count + size
        
                #embed words in sentence
                embedding.embed(sentence)
            
                for j in range(size):
                    A = A + sentence[j].embedding
            
            else:
                sentence = Sentence(f1[3*(i+1)])
                size = len(sentence)
                if size>430:
                    print('bad sentences')
                    return []
                 
                token_count = token_count + size
                
                #embed words in sentence
                embedding.embed(sentence) 
            
                for j in range(size):
                    A = A + sentence[j].embedding
 
                sentence = Sentence(f1[3*(i+1)+1])
                size = len(sentence)
                if size>430:
                    print('bad sentences')
                    return []
                 
                token_count = token_count + size
                
                #embed words in sentence
                embedding.embed(sentence) 
            
                for j in range(size):
                    A = A + sentence[j].embedding
                     
                sentence = Sentence(f1[3*(i+1)+2])
                size = len(sentence)
                if size>430:
                    print('bad sentences')
                    return []
 
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
            if size<430:
                
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
vec_number = 50

for i in range(10):        
    if i!=0 and i!=8:
        random.shuffle(geniss[i])
        print('topic_'+str(i)+'started...')
        for j in range(vec_number):
            print(str(j)+'.'+geniss[i][j])
            vectors[i].append((geniss[i][j],bert_doc_embed(geniss[i][j])))
            
        print('topic_'+str(i)+'completed')
            
file = open('/Users/Ani/Desktop/bert-topic-vecs.pkl','wb')
pickle.dump(vectors,file)
file.close()            
            
    