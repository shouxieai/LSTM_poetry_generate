import numpy as np
from gensim.models.word2vec import Word2Vec
import pickle
from torch.utils.data import Dataset , DataLoader
import os
import torch
import torch.nn as nn


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

class Mymodel(nn.Module):

    def __init__(self,embedding_num,hidden_num,word_size):
        super().__init__()

        self.embedding_num = embedding_num
        self.hidden_num = hidden_num
        self.word_size = word_size

        self.lstm = nn.LSTM(input_size = embedding_num,hidden_size = hidden_num, batch_first = True,num_layers = 2, bidirectional = False)
        self.dropout = nn.Dropout(0.3)  # 有了随机失活, 生成的古诗就不会唯一了
        self.flatten = nn.Flatten(0,1)
        self.linear = nn.Linear(hidden_num,word_size)
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, xs_embedding):

        hidden,(h_0,c_0) = self.lstm(xs_embedding)
        hidden_drop = self.dropout(hidden)
        flatten_hidden = self.flatten(hidden_drop)
        pre = self.linear(flatten_hidden)

        return pre



if __name__ == "__main__" :

    batch_size = 5
    all_data ,(w1,word_2_index,index_2_word) = train_vec()
    dataset = MyDataset(all_data,w1,word_2_index)
    dataloader = DataLoader(dataset,batch_size = batch_size,shuffle = True)

    hidden_num = 51
    lr = 0.001

    epochs = 1000

    word_size , embedding_num = w1.shape

    model = Mymodel(embedding_num,hidden_num,word_size)
    optimizer = torch.optim.AdamW(model.parameters(),lr = lr)

    for e in range(epochs):
        for batch_index,(xs_embedding ,ys_index) in enumerate(dataloader):

            pre = model(xs_embedding)
            loss = model.cross_entropy(pre,ys_index.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_index % 30 == 0 :
                print(f"loss:{loss:.3f}")


    pass


