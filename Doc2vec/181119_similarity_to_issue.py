#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import glob
import os
import random
import logging
import pandas as pd
import numpy as np
import pickle as pk

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk import tokenize
from zipfile import ZipFile

#setting the working directory
os.chdir('/home/jcai/geometry_of_law/')
#loading the model
model = Doc2Vec.load('/home/jcai/geometry_of_law/doc2vec_v50k_d200_shuffled_opinion/ALL_opinion.d2v')

#calculate and export similarity to "regulation", "Privacy", "Labor"
list_of_issues = ["criminal-appeals", "civil-rights", "first-admendment","due-process", "privacy", "labor", "regulation"]

list_of_names = ["criminal appeals", "civil rights", "first admendment","due process", "privacy", "labor", "regulation"]

issue_dict = dict(zip(list_of_issues,list_of_names))

number_of_opinions = model.docvecs.count

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
        	top_list = model.docvecs.most_similar(positive=[vec_list[i]],topn=number_of_opinions)
        	id_list = [item[0] for item in top_list]
        	caseid_list = [item[0].partition('_')[0] for item in top_list]
        	similarity = [item[1] for item in top_list]

        	df = pd.DataFrame([id_list,caseid_list,similarity])
        	df = df.T

        	similarity_string = 'similarity_to_' + issue + '_' + name[i]
        	df.columns = ['id_list','caseid_list', similarity_string]

        	path = 'similarity_to_issues/'
        	path_name = path + str(issue)+ '_'+ name[i] + '.csv'
        	df.to_csv(path_name, index=False)
        	print(path_name +' is saved')
        except Exception as e:
            print(repr(e))
            print (issue+' '+'is not saved, there was an error')
            pass