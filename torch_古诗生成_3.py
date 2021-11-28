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


def train_vec(vector_size=128, split_file="split_7.txt", org_file="poetry_7.txt", train_num=6000):
    param_file = "word_vec.pkl"
    org_data = open(org_file, "r", encoding="utf-8").read().split("\n")[:train_num]
    if os.path.exists(split_file):
        all_data_split = open(split_file, "r", encoding="utf-8").read().split("\n")[:train_num]
    else:
        all_data_split = split_text().split("\n")[:train_num]

    if os.path.exists(param_file):
        return org_data, pickle.load(open(param_file, "rb"))

    models = Word2Vec(all_data_split, vector_size=vector_size, workers=7, min_count=1)
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
    def __init__(self, params):
        super().__init__()
        self.all_data, (self.w1, self.word_2_index, self.index_2_word) = train_vec(vector_size=params["embedding_num"],
                                                                                   train_num=params["train_num"])

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.hidden_num = params["hidden_num"]
        self.batch_size = params["batch_size"]
        self.epochs = params["epochs"]
        self.lr = params["lr"]
        self.optimizer = params["optimizer"]
        self.word_size, self.embedding_num = self.w1.shape

        self.lstm = nn.LSTM(input_size=self.embedding_num, hidden_size=self.hidden_num, batch_first=True, num_layers=2,
                            bidirectional=False)
        self.dropout = nn.Dropout(0.3)  # 古诗不具有唯一性
        self.flatten = nn.Flatten(0, 1)
        self.linear = nn.Linear(self.hidden_num, self.word_size)
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

    def to_train(self):
        model_result_file = "Poetry_Model_lstm_model.pkl"
        if os.path.exists(model_result_file):
            return pickle.load(open(model_result_file, "rb"))
        dataset = Poetry_Dataset(self.w1, self.word_2_index, self.all_data)
        dataloader = DataLoader(dataset, self.batch_size)

        optimizer = self.optimizer(self.parameters(), self.lr)
        self = self.to(self.device)
        for e in range(self.epochs):
            for batch_index, (batch_x_embedding, batch_y_index) in enumerate(dataloader):
                self.train()
                batch_x_embedding = batch_x_embedding.to(self.device)
                batch_y_index = batch_y_index.to(self.device)

                pre, _ = self(batch_x_embedding)
                loss = self.cross_entropy(pre, batch_y_index.reshape(-1))

                loss.backward()  # 梯度反传 , 梯度累加, 但梯度并不更新, 梯度是由优化器更新的
                optimizer.step()  # 使用优化器更新梯度
                optimizer.zero_grad()  # 梯度清零

                if batch_index % 100 == 0:
                    print(f"loss:{loss:.3f}")
                    self.generate_poetry_auto()
        pickle.dump(self, open(model_result_file, "wb"))
        return self

    def generate_poetry_auto(self):
        # self.eval()
        result = ""
        word_index = np.random.randint(0, self.word_size, 1)[0]

        result += self.index_2_word[word_index]
        h_0 = torch.tensor(np.zeros((2, 1, self.hidden_num), dtype=np.float32))
        c_0 = torch.tensor(np.zeros((2, 1, self.hidden_num), dtype=np.float32))

        for i in range(31):
            word_embedding = torch.tensor(self.w1[word_index][None][None])
            pre, (h_0, c_0) = self(word_embedding, h_0, c_0)
            word_index = int(torch.argmax(pre))
            result += self.index_2_word[word_index]

        print(result)

    def generate_poetry_acrostic(self):

        while True:

            input_text = input("请输入四个汉字：")[:4]
            if input_text == "":
                self.generate_poetry_auto()
            else:

                result = ""
                punctuation_list = ["，", "。", "，", "。"]
                for i in range(4):

                    h_0 = torch.tensor(np.zeros((2, 1, self.hidden_num), dtype=np.float32))
                    c_0 = torch.tensor(np.zeros((2, 1, self.hidden_num), dtype=np.float32))
                    word = input_text[i]
                    try:
                        word_index = self.word_2_index[word]
                    except:
                        word_index = np.random.randint(0, self.word_size, 1)[0]
                        word = self.index_2_word[word_index]
                    result += word

                    for j in range(6):
                        word_index = self.word_2_index[word]
                        word_embedding = torch.tensor(self.w1[word_index][None][None])
                        pre, (h_0, c_0) = model(word_embedding, h_0, c_0)
                        word = self.index_2_word[int(torch.argmax(pre))]
                        result += word
                    result += punctuation_list[i]
                print(result)


if __name__ == "__main__":
    # ---------------------------------  个性化参数  --------------------------------------
    params = {
        "batch_size": 32,  # batch大小
        "epochs": 1000,  # epoch大小
        "lr": 0.003,  # 学习率
        "hidden_num": 64,  # 隐层大小
        "embedding_num": 128,  # 词向量维度
        "train_num": 1000,  # 训练的故事数量, 七言古诗:0~6290, 五言古诗:0~2929
        "optimizer": torch.optim.AdamW,  # 优化器 , 注意不要加括号
        "batch_num_test": 100,  # 多少个batch 打印一首古诗进行效果测试
    }

    model = Poetry_Model_lstm(params)  # 模型定义
    model = model.to_train()  # 模型训练
    model.generate_poetry_acrostic()  # 测试藏头诗



