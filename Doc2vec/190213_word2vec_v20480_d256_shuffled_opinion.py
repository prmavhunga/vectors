# coding: utf-8

# loads packages
import glob
import os
import random
import logging
import pandas
import numpy
import pickle
import gensim
import operator
import re
from random import shuffle


from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from typing import Iterable

from gensim.models.word2vec import Word2Vec
from nltk import tokenize, casual_tokenize
import nltk 
from zipfile import ZipFile
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktTrainer
from nltk.corpus import stopwords

print('load packages done')

# defines vocabulary size and vector dimension
voc_size = 20480
vec_dim  = 256
model_name = 'cc'
directory_name = 'word2vec_v20480_d256_shuffled_opinion'


# configues loggin
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO, filename=directory_name+'/'+model_name+'.log')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

# sets working directory
wd = '/home/jcai/geometry_of_law'
os.chdir(wd)
print(wd + ' is wd')

#read zipfiles
zfile_cc = ZipFile('/data/Data/Circuit_Courts/circuit-cases/sentences.zip', allowZip64 = True)
items_cc = [("cc", x) for x in zfile_cc.namelist()]
items_cc = [x for x in items_cc if '.txt' in x[1]]
items_cc = [x for x in items_cc if int(x[1].split("/")[1].split("_")[1]) in range(1970,2006)]

#zfile_sc = ZipFile('/data/Data/Supreme_Court_Cases/SC_Cases_1880_2016.zip', allowZip64 = True)
#items_sc = [("sc", x) for x in zfile_sc.namelist()]
#items_sc = [x for x in items_sc if '.txt' in x[1]]

#items_cc.extend(items_sc)
shuffle(items_cc)

#removing stopword
stopwords_year = [str(x) for x in list(range(1970,2005))]

#get judgenames
judge_bio = pandas.read_stata("/data/Data/Judge-Bios/judgebios/JudgesBioReshaped_TOUSE.dta")
#stopphrase_judge_songer_name = list(judge_bio["songername"])
#stopphrase_wo_middle = [judge_bio['judgefirstname'][x]+" "+judge_bio['judgelastname'][x] for x in judge_bio.index]
#stopphrase_w_middle = [judge_bio['judgefirstname'][x]+" "+judge_bio['judgemiddlename'][x]+" "+judge_bio['judgelastname'][x] for x in judge_bio.index]
stopwords_judge_name = list(judge_bio['judgefirstname']) + list(judge_bio['judgelastname'])

#lists_of_stopwords = stopwords_judge_songer_name + stopwords_wo_middle + stopwords_w_middle
#lists_of_stopwords = [x.lower() for x in lists_of_stopwords]
#stop_regex = "|".join(lists_of_stopwords)
#defines a tokenizer that does not save punctuation and convert everything to lower case
#also removes stopwords
stop = stopwords.words('english') + stopwords_year + stopwords_judge_name
stop = [x.lower() for x in stop]
stop = set(stop)

def tokenize_no_punct_all_lower(txt):
    txt_tokenize = casual_tokenize(txt,preserve_case=False,strip_handles=True)
    txt_tokenize = [word for word in txt_tokenize if re.sub(r"\-", "", word).isalpha()]
    txt_tokenize = [word for word in txt_tokenize if word not in stop]
    return txt_tokenize

#defines iterator that splits by lines
def document_iterator_lines(list_of_docnames):
    for count, item in enumerate(list_of_docnames):
        fname = item[1].split('/')[-1]
        if not count % 50000:
            print('%d files processed' % count)   

        if item[0] == "cc":
            txt = zfile_cc.open(item[1],"r").read()
        else:
            txt = zfile_sc.open(item[1],"r").read()
        
        txt = txt.decode("utf-8")
        tokens = tokenize_no_punct_all_lower(txt)
        tokens_tagged = nltk.pos_tag(tokens,tagset='universal')
        tokens = [i[0] for i in tokens_tagged if i[1] in ["VERB","NOUN","ADJ","ADV"] ]
        yield tokens

#wraps generator in an iterable object
class doc_iter(object):
    def __init__(self, fname):
        self.fname = fname 
        
    def __iter__(self):
        for doc in document_iterator_lines(self.fname):
            yield doc

#docs
sentences = doc_iter(items_cc)

#loads word frequency dict and retrieve the frequency of the 50000th word
#word_freq_dict_min_50 = pickle.load(open('/data_alt/WorkData/arianna/word_embeddings/dictionaries/dictionary_1970.p','rb'))
#sorted_words = sorted(word_freq_dict_min_50.items(), key=operator.itemgetter(1), reverse=True)
#min_count_for_vocab = sorted_words[voc_size-1][1]
#print('word_frequency is ' + str(min_count_for_vocab))

#original model
model = Word2Vec(max_vocab_size=voc_size, window=8, size=vec_dim, sample=1e-4, negative=5, workers=16)
# builds the vocabulary
model.build_vocab(sentences)

# trains the embeddings
model.train(sentences, total_examples=model.corpus_count, epochs=16)

# saves the model
model.save(directory_name+'/'+model_name+'.w2v')

# saves word counts as pickle
dictionary = {word: vocab.count for (word, vocab) in model.wv.vocab.items()}
pickle.dump(dictionary, open(directory_name+'/'+model_name+'.p', 'wb' ))

print('model is saved')
    
model.delete_temporary_training_data()


