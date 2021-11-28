import numpy as np
from gensim.models.word2vec import Word2Vec
import pickle
from torch.utils.data import Dataset , DataLoader
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


class MyDataset(Dataset):

    # 加载所有的数据
    # 存储和初始化一些变量
    def __init__(self,all_data,w1,word_2_index):
        self.w1 = w1
        self.word_2_index = word_2_index
        self.all_data = all_data

    # 获取一条数据, 并作处理
    def __getitem__(self,index):
        a_poetry_words = self.all_data[index]
        a_poetry_index = [self.word_2_index[word] for word in a_poetry_words]

        xs_index = a_poetry_index[:-1]
        ys_index = a_poetry_index[1:]

        xs_embedding = self.w1[xs_index]

        return xs_embedding, np.array(ys_index).astype(np.int64)

    #  获取数据的总长度
    def __len__(self):
        return len(all_data)



if __name__ == "__main__" :

    batch_size = 5
    all_data ,(w1,word_2_index,index_2_word) = train_vec()
    dataset = MyDataset(all_data,w1,word_2_index)
    dataloader = DataLoader(dataset,batch_size = batch_size,shuffle = True)

    for xs_embedding ,ys_index in dataloader:
        print("")

    pass


