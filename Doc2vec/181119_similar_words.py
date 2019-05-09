#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import glob
import os
import random
import logging
import pandas as pd
import numpy as np
import pickle as pk
import sys

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk import tokenize
from zipfile import ZipFile

#setting the working directory
os.chdir('/home/jcai/geometry_of_law/doc2vec_v50k_d200_shuffled_opinion')
path =  '/home/jcai/geometry_of_law/similar_words/'
#loading the model
model = Doc2Vec.load('/home/jcai/geometry_of_law/doc2vec_v50k_d200_shuffled_opinion/ALL_opinion.d2v')

#calculate and export similarity to "regulation", "Privacy", "Labor"
list_of_issues = ["criminal-appeals", "civil-rights", "first-admendment","due-process", "privacy", "labor", "regulation"]

list_of_names = ["criminal appeals", "civil rights", "first admendment","due process", "privacy", "labor", "regulation"]

issue_dict = dict(zip(list_of_issues,list_of_names))

for issue in issue_dict.keys():
    word_list = issue_dict[issue].split(" ")
    #simply mean the vector
    add_vec_list = []
    for word in word_list:
        try:
            add_vec_list.append(model[word])
        except:
            print(word + " is not in model")
            pass

    add_vec = np.mean(add_vec_list, axis=0)
    #infer vector
    inferred_vec = model.infer_vector(issue)
    #find similar words
    vec_list = [add_vec,inferred_vec]
    for i in range(len(vec_list)):
        name = ["add_vec","inferred_vec"]
        try:
            df = pd.DataFrame(model.wv.similar_by_vector(vec_list[i], topn=100))
            df.columns = ['word','similarity']
            path_name = path + str(issue)+ '_'+ name[i] + '_word_sim.csv'
            df.to_csv(path_name, index=False)
            print(path_name +' is saved')
        except Exception as e:
            print(repr(e))
            print (issue+' '+'is not saved, there was an error')
            pass