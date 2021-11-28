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

    split_all_data = open(split_file, "r", encoding="utf-8").read().split("\n")[:1000]
    org_data = open(org_file, "r", encoding="utf-8").read().split("\n")[:1000]

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

    def forward(self, xs_embedding,h_0 = None,c_0 = None):
        xs_embedding = xs_embedding.to(device)
        if h_0 == None or c_0 == None:
            h_0 = torch.tensor(np.zeros((2,xs_embedding.shape[0],self.hidden_num),np.float32))
            c_0 = torch.tensor(np.zeros((2,xs_embedding.shape[0],self.hidden_num),np.float32))
        h_0 = h_0.to(device)
        c_0 = c_0.to(device)
        hidden,(h_0,c_0) = self.lstm(xs_embedding,(h_0,c_0))
        hidden_drop = self.dropout(hidden)
        flatten_hidden = self.flatten(hidden_drop)
        pre = self.linear(flatten_hidden)

        return pre,(h_0,c_0)

def generate_poetry_auto():

    result = ""
    word_index = np.random.randint(0,word_size,1)[0]
    result += index_2_word[word_index]

    h_0 = torch.tensor(np.zeros((2, 1, hidden_num), np.float32))
    c_0 = torch.tensor(np.zeros((2, 1, hidden_num), np.float32))
    for i in range(31):
        # word_embedding = torch.tensor(w1[word_index].reshape(1,1,-1))
        word_embedding = torch.tensor(w1[word_index][None][None])
        pre,(h_0,c_0) = model(word_embedding,h_0,c_0)
        word_index = int(torch.argmax(pre))
        result += index_2_word[word_index]
    print(result)

if __name__ == "__main__" :

    device = "cuda" if torch.cuda.is_available() else "cpu"

    batch_size = 64
    all_data ,(w1,word_2_index,index_2_word) = train_vec()
    dataset = MyDataset(all_data,w1,word_2_index)
    dataloader = DataLoader(dataset,batch_size = batch_size,shuffle = True)

    hidden_num = 128
    lr = 0.007

    epochs = 1000

    word_size , embedding_num = w1.shape

    model = Mymodel(embedding_num,hidden_num,word_size)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(),lr = lr)

    for e in range(epochs):
        for batch_index,(xs_embedding ,ys_index) in enumerate(dataloader):
            xs_embedding = xs_embedding.to(device)
            ys_index = ys_index.to(device)

            pre,_ = model(xs_embedding)
            loss = model.cross_entropy(pre,ys_index.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_index % 100 == 0 :
                print(f"loss:{loss:.3f}")
                generate_poetry_auto()



