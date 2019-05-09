#!/usr/bin/env python
# coding: utf-8

# In[1]:


import glob
import os
import random
import logging
import pandas as pd
import numpy as np
import pickle as pk
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from nltk import ngrams
import itertools
import re
from multiprocessing import Process, Pool, Manager
import multiprocessing
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


# # loading metadata

# In[2]:


outcome_data = pd.read_stata('/data/Projects/ornelie_dlchen/HundredPercentData_Bloomberg/DTAs/TOUSE/BloombergCASELEVEL_Touse.dta')


# In[3]:


outcome_data = outcome_data[['caseid', 'Circuit', 'citation','dissentdummy','reversedummy', 'affirmdummy']]


# In[4]:


master_dataframe = pd.read_csv("/home/jcai/geometry_of_law/Encyclopedia Entry/master_dataframe.csv")


# In[5]:


model = Doc2Vec.load('/home/jcai/geometry_of_law/doc2vec_v50k_d200_shuffled_opinion/ALL_opinion.d2v')
set_of_doc_names = set(model.docvecs.doctags)
master_dataframe = master_dataframe[master_dataframe['docname'].isin(set_of_doc_names)]


# # load doc2vec vectors

# In[6]:


master_dataframe['d2v'] = [model[master_dataframe['docname'][i]] for i in master_dataframe.index]


# # load and flter n_grams according to grammar

# In[7]:


grammar = set((("A","N"),("N","N"),("V","N"),("V","V"),("N","V"),("V","P")))


# In[8]:


def filter_grammar(grammar,n_gram):
    if tuple([x[1] for x in n_gram]) in grammar:
        str_n_gram = ' '.join([x[0] for x in n_gram])
        return str_n_gram
    else:
        return None


# In[9]:


def load_n_gram(indx):
    year = str(master_dataframe['year'][indx])
    docname = master_dataframe['docname'][indx][:-4]
    year_dir = "/home/weilu/GST_data/final_20190123/final_tokens_snowball/"+year+"/"
    
    doc_tokens = []
    try:
        path_to_n_gram = year_dir+docname+".pkl"
        doc_tokens = pk.load(open(path_to_n_gram,"rb"))
    except:
        try:
            for f in os.listdir(year_dir):
                if re.match(docname.split("_")[0] + "_" +docname.split("_")[1]+"*",f):
                    doc_tokens = pk.load(open(year_dir+f,"rb"))
        except Exception as e:
            print(e)
    
    sent_n_grams = list(itertools.chain.from_iterable([list(ngrams(filter(None,sent),2)) for sent in doc_tokens]))
    n_grams = list(filter(None,[filter_grammar(grammar, n_gram) for n_gram in sent_n_grams]))
    return n_grams


# In[10]:


manager = Manager()
pool = Pool(processes=multiprocessing.cpu_count()-1)
print("CPU count is " + str(multiprocessing.cpu_count()))
result = pool.map(load_n_gram, master_dataframe.index)
pool.close()
pool.join()


# In[11]:


master_dataframe['n_grams'] = result


# # build predictive models

# In[12]:


idx = [i for i in master_dataframe.index if master_dataframe["n_grams"][i] != []]


# In[13]:


master_dataframe = master_dataframe.iloc[idx,:]


# In[14]:


master_dataframe["caseid"] = [x.split("_")[0] for x in master_dataframe.copy()['docname']]


# ### convert list of n_gram to a frequency dictionary 

# In[15]:


from collections import Counter


# In[16]:


manager = Manager()
pool = Pool(processes=multiprocessing.cpu_count()-1)
print("CPU count is " + str(multiprocessing.cpu_count()))
result = pool.map(Counter, master_dataframe["n_grams"])
pool.close()
pool.join()


# In[17]:


master_dataframe["n_gram_freq"] = result


# In[18]:


merged = master_dataframe.copy().merge(outcome_data, how = "left",on = "caseid")


# In[19]:


merged = merged[['circuit','d2v','caseid','n_gram_freq','dissentdummy', 'reversedummy', 'affirmdummy']]


# ## featurize

# In[24]:


from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer


# In[21]:


dictvectorizer = DictVectorizer(sparse=True)


# In[22]:


n_gram_X = dictvectorizer.fit_transform(merged['n_gram_freq'])


# In[23]:


n_gram_X.size


# ### we have 242596955 2_grams

# In[25]:


tfidf_vectorizer = TfidfTransformer(use_idf=True)


# In[26]:


tfidf_vectorizer.fit(n_gram_X)


# In[29]:


idf = tfidf_vectorizer.idf_


# # take those with idf < 6.74 (about 27000) features

# In[54]:


thres = np.percentile(idf, .1)


# In[55]:


feature_idx = [i for i in range(len(idf)) if idf[i] < thres]


# In[59]:


feature_names_selected = [dictvectorizer.feature_names_[i] for i in feature_idx]


# In[66]:


n_gram_X_csc = n_gram_X.tocsc(copy=True)


# In[70]:


n_gram_X_selected = n_gram_X_csc[:,feature_idx]


# In[72]:


n_gram_X_selected = n_gram_X_selected.tocsr()


# In[82]:


reverse_y = merged["reversedummy"]


# ## need to do feature selection for n_gram

# ## calculate term document frequecy

# In[73]:


from sklearn.feature_extraction.text import TfidfTransformer


# In[94]:


n_gram_X_tfidf = tfidf_vectorizer.fit_transform(X=n_gram_X_selected)


# ## d2v features

# In[138]:


d2v_X = np.asarray(*merged["d2v"])


# In[140]:


merged["d2v"][0].shape


# In[ ]:




# In[126]:


n_gram_X_tfidf.shape


# ## logistic regression

# In[78]:


from sklearn.linear_model import LogisticRegression


# In[96]:


n_gram_X_train, n_gram_X_test, n_gram_reverse_y_train, n_gram_reverse_y_test = train_test_split(n_gram_X_tfidf, reverse_y, test_size=0.1, random_state=42)


# In[121]:


d2v_X_train, d2v_X_test, d2v_reverse_y_train, d2v_reverse_y_test = train_test_split(d2v_X, reverse_y, test_size=0.1, random_state=42)


# In[122]:


d2v_log = LogisticRegression()


# In[123]:


d2v_log.fit(d2v_X_train,d2v_reverse_y_train)


# In[ ]:


log.fit(n_gram_X_train, affirm_y_train)


# In[ ]:




