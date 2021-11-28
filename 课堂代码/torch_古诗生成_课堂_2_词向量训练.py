import numpy as np
from gensim.models.word2vec import Word2Vec
import pickle
import os

def spilt_poetry(file = "poetry_7.txt"):
    all_data = open(file,"r",encoding = "utf-8").read()
    all_data_split = " ".join(all_data)
    with open("split.txt","w",encoding = 'utf-8') as f:
        f.write(all_data_split)

def train_vec(split_file = "split.txt",org_file = "poetry_7.txt"):
    vec_params_file = "vec_params.pkl"
    if os.path.exists(split_file) == False:
        spilt_poetry()

    split_all_data = open(split_file, "r", encoding="utf-8").read().split("\n")
    org_data = open(org_file, "r", encoding="utf-8").read().split("\n")

    if os.path.exists(vec_params_file):
        return org_data,pickle.load(open(vec_params_file,"rb"))

    model = Word2Vec(split_all_data,vector_size=107,min_count = 1,workers = 6)
    pickle.dump((model.syn1neg,model.wv.key_to_index,model.wv.index_to_key),open(vec_params_file,"wb"))

    return org_data , (model.syn1neg,model.wv.key_to_index,model.wv.index_to_key)



if __name__ == "__main__" :

    all_data ,(w1,word_2_index,index_2_word) = train_vec()
    pass


