import os
import numpy as np
import pickle
import torch
import torch.nn as nn
from gensim.models.word2vec import Word2Vec
from torch.utils.data import Dataset, DataLoader


def split_text(file="poetry_7.txt", train_num=6000):
    all_data = open(file, "r", encoding="utf-8").read()
    with open("split_7.txt", "w", encoding="utf-8") as f:
        split_data = " ".join(all_data)
        f.write(split_data)
    return split_data[:train_num * 64]


def train_vec(split_file="split_7.txt", org_file="poetry_7.txt", train_num=6000):
    param_file = "word_vec.pkl"
    org_data = open(org_file, "r", encoding="utf-8").read().split("\n")[:train_num]
    if os.path.exists(split_file):
        all_data_split = open(split_file, "r", encoding="utf-8").read().split("\n")[:train_num]
    else:
        all_data_split = split_text().split("\n")[:train_num]

    if os.path.exists(param_file):
        return org_data, pickle.load(open(param_file, "rb"))

    models = Word2Vec(all_data_split, vector_size=128, workers=7, min_count=1)
    pickle.dump([models.syn1neg, models.wv.key_to_index, models.wv.index_to_key], open(param_file, "wb"))
    return org_data, (models.syn1neg, models.wv.key_to_index, models.wv.index_to_key)


class Poetry_Dataset(Dataset):
    def __init__(self, w1, word_2_index, all_data):
        self.w1 = w1
        self.word_2_index = word_2_index
        self.all_data = all_data

    def __getitem__(self, index):
        a_poetry = self.all_data[index]

        a_poetry_index = [self.word_2_index[i] for i in a_poetry]
        xs = a_poetry_index[:-1]
        ys = a_poetry_index[1:]
        xs_embedding = self.w1[xs]

        return xs_embedding, np.array(ys).astype(np.int64)

    def __len__(self):
        return len(self.all_data)


class Poetry_Model_lstm(nn.Module):
    def __init__(self, hidden_num, word_size, embedding_num):
        super().__init__()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.hidden_num = hidden_num

        self.lstm = nn.LSTM(input_size=embedding_num, hidden_size=hidden_num, batch_first=True, num_layers=2,
                            bidirectional=False)
        self.dropout = nn.Dropout(0.3)
        self.flatten = nn.Flatten(0, 1)
        self.linear = nn.Linear(hidden_num, word_size)
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, xs_embedding, h_0=None, c_0=None):
        if h_0 == None or c_0 == None:
            h_0 = torch.tensor(np.zeros((2, xs_embedding.shape[0], self.hidden_num), dtype=np.float32))
            c_0 = torch.tensor(np.zeros((2, xs_embedding.shape[0], self.hidden_num), dtype=np.float32))
        h_0 = h_0.to(self.device)
        c_0 = c_0.to(self.device)
        xs_embedding = xs_embedding.to(self.device)
        hidden, (h_0, c_0) = self.lstm(xs_embedding, (h_0, c_0))
        hidden_drop = self.dropout(hidden)
        hidden_flatten = self.flatten(hidden_drop)
        pre = self.linear(hidden_flatten)

        return pre, (h_0, c_0)


def generate_poetry_auto():
    result = ""
    word_index = np.random.randint(0, word_size, 1)[0]

    result += index_2_word[word_index]
    h_0 = torch.tensor(np.zeros((2, 1, hidden_num), dtype=np.float32))
    c_0 = torch.tensor(np.zeros((2, 1, hidden_num), dtype=np.float32))

    for i in range(31):
        word_embedding = torch.tensor(w1[word_index][None][None])
        pre, (h_0, c_0) = model(word_embedding, h_0, c_0)
        word_index = int(torch.argmax(pre))
        result += index_2_word[word_index]

    return result


def generate_poetry_acrostic():
    input_text = input("请输入四个汉字：")[:4]
    result = ""
    punctuation_list = ["，", "。", "，", "。"]
    for i in range(4):
        result += input_text[i]
        h_0 = torch.tensor(np.zeros((2, 1, hidden_num), dtype=np.float32))
        c_0 = torch.tensor(np.zeros((2, 1, hidden_num), dtype=np.float32))
        word = input_text[i]
        for j in range(6):
            word_index = word_2_index[word]
            word_embedding = torch.tensor(w1[word_index][None][None])
            pre , (h_0,c_0) = model(word_embedding,h_0,c_0)
            word = word_2_index[int(torch.argmax(pre))]
            result += word

    return result



if __name__ == "__main__":

    all_data, (w1, word_2_index, index_2_word) = train_vec(train_num=300)

    batch_size = 32
    epochs = 1000
    lr = 0.01
    hidden_num = 128
    word_size, embedding_num = w1.shape

    dataset = Poetry_Dataset(w1, word_2_index, all_data)
    dataloader = DataLoader(dataset, batch_size)

    model = Poetry_Model_lstm(hidden_num, word_size, embedding_num)
    model = model.to(model.device)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    for e in range(epochs):
        for batch_index, (batch_x_embedding, batch_y_index) in enumerate(dataloader):
            model.train()
            batch_x_embedding = batch_x_embedding.to(model.device)
            batch_y_index = batch_y_index.to(model.device)

            pre, _ = model(batch_x_embedding)
            loss = model.cross_entropy(pre, batch_y_index.reshape(-1))

            loss.backward()  # 梯度反传 , 梯度累加, 但梯度并不更新, 梯度是由优化器更新的
            optimizer.step()  # 使用优化器更新梯度
            optimizer.zero_grad()  # 梯度清零

            if batch_index % 100 == 0:
                # model.eval()
                print(f"loss:{loss:.3f}")
                print(generate_poetry_auto())

    print("")
