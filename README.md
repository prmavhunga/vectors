# case_vectors
Code base for making case vectors and using them for empirical analysis.

# descriptions
190106_doc2vec_v50k_d200_shuffled_sentences.py

This script trains and saves a d2v model using both supreme court and circuit court cases.  All the scripts using d2v vectors as features start with the vectors saved by this script.

181111_making_dictionary.ipynb

Saves the mapping from metadata characteristics to average vectors of documents with such characteristics as dictionaries.  Vectors come from the d2v model saved from the previous script.  Metadata from from "/home/jcai/geometry_of_law/Encyclopedia Entry/master_dataframe.csv" -- a version of the bloomberg data. 

181111_making_vector_atrribute_dictionary.ipynb

Saves the mapping from document-id to metadata characteristics.

181111_dis_data_generation.ipynb

Using the above data we calculate the distance between a case vector to the average and regress the distance on affirmed/disent.

The rest of the scripts are mostly ad-hoc tasks done using the d2v model saved in the first script.  



