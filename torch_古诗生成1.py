import numpy as np
from gensim.models.word2vec import Word2Vec
from torch.utils.data import Dataset, DataLoader
import pickle,os,random
import torch
import torch.nn as nn

def split(file = "poetry_5.txt"):
    all_data = open(file , "r",encoding="utf-8").read()

    with open("split.txt","w",encoding = "utf-8") as f:
        f.write(" ".join(all_data))
    return all_data.split("\n")

def train_word_2_vec(file = "split.txt"):
    params_file = "params.pkl"
    if os.path.exists(file):
        all_data = open(file,"r",encoding = "utf-8").read().split("\n")
    else :
        all_data  = split()
    if os.path.exists(params_file):
        return ["".join(i.split(" ")) for i in all_data],pickle.load(open(params_file,"rb"))
    else:
        model = Word2Vec(all_data,vector_size = 107,min_count = 1,workers =5,window = 4,hs = 0,sg = 0)
        pickle.dump([model.syn1neg,model.wv.key_to_index,model.wv.index_to_key],open(params_file,"wb"))
        return ["".join(i.split(" ")) for i in all_data],(model.syn1neg,model.wv.key_to_index,model.wv.index_to_key)

class MyDataset:
    def __init__(self,all_data ,w1,word_2_index,batch_size , shuffle = True):
        self.w1 = w1
        self.all_data = all_data
        self.word_2_index = word_2_index
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        return MyDataLoader(self)

    def __len__(self):
        return len(self.all_data)

class MyDataLoader:
    def __init__(self,dataset):
        self.dataset = dataset
        self.cursor = 0
        self.end = len(self.dataset)
        if self.dataset.shuffle :
            self.index = random.sample(range(0,self.end),self.end)
        else:
            self.index = np.arange(0,self.end)
        self.max_b = int(np.ceil(self.end/self.dataset.batch_size))

    def __next__(self):
        if self.cursor >= self.max_b:
            raise StopIteration()

        batch_poetry = [self.dataset.all_data[i] for i in self.index[self.cursor * self.dataset.batch_size : (self.cursor + 1) * self.dataset.batch_size] ]
        words_index = [[word_2_index[j] for j in i] for i in batch_poetry]
        xs_embedding = np.array([self.dataset.w1[i[:-1]] for i in words_index])
        ys_label = np.array([i[1:] for i in words_index])
        self.cursor += 1
        return torch.tensor(xs_embedding),torch.tensor(ys_label)

class Mymodel(nn.Module):
    def __init__(self,hidden_num,embedding_num,word_size):
        super().__init__()
        self.hidden_num = hidden_num
        self.embedding_num = embedding_num
        self.word_size = word_size

        self.lstm = nn.LSTM(input_size = self.embedding_num,hidden_size = self.hidden_num,num_layers = 3,bidirectional = False,batch_first = True)
        self.linear = nn.Linear(1 * self.hidden_num,self.word_size)
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self,xs_embedding,h_0 = None, c_0 = None):
        xs_embedding = xs_embedding.to(device)
        if h_0 == None or c_0 == None:
            h_0 = torch.tensor(np.zeros((3 * 1, xs_embedding.shape[0], self.hidden_num),dtype = np.float32))
            c_0 = torch.tensor(np.zeros((3 * 1, xs_embedding.shape[0], self.hidden_num),dtype = np.float32))
        h_0 = h_0.to(device)
        c_0 = c_0.to(device)
        h,(h_0,c_0) = self.lstm(xs_embedding,(h_0,c_0))
        p = self.linear(h.reshape(-1, self.hidden_num))

        return p ,( h_0,c_0)


def generate_poetry():
    result = ""
    word_index = np.random.randint(0,word_size,1)[0]

    result += index_2_word[word_index]

    h_0 = torch.tensor(np.zeros((3 * 1, 1, model.hidden_num), dtype=np.float32))
    c_0 = torch.tensor(np.zeros((3 * 1, 1, model.hidden_num), dtype=np.float32))

    for i in range(23):
        word_embedding = torch.tensor(w1[word_index].reshape(1, 1, -1))
        p,(h_0,c_0) = model(word_embedding,h_0,c_0)
        word_index = int(torch.argmax(p))
        result += index_2_word[word_index]
    return result


if __name__ == "__main__":
    all_data , (w1 , word_2_index, index_2_word) = train_word_2_vec()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    epochs = 1000
    hidden_num = 128
    lr = 0.007
    word_size , embedding_num = w1.shape
    batch_size = 100

    dataset = MyDataset(all_data,w1,word_2_index,batch_size)
    model = Mymodel(hidden_num,embedding_num,word_size)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(),lr = lr)

    for e in range(epochs):
        for batch_index,(batch_embeddings,batch_labels) in enumerate(dataset):
            batch_embeddings = batch_embeddings.to(device)
            batch_labels = batch_labels.to(device)
            p,_ = model(batch_embeddings)
            loss = model.cross_entropy.forward(p,batch_labels.reshape(-1).long())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_index % 300 == 0:

                print(f"loss:{loss:.3f}")
                print(generate_poetry())