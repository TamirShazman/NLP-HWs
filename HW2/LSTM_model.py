from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from copy import copy
from sklearn.metrics import f1_score
from nltk.stem import PorterStemmer

from dl_models import FocalLoss

class LSTMDataset(Dataset):
    def __init__(self, sentences, labels, sen_len):
        self.sentences = sentences
        self.labels = labels
        self.sen_len = sen_len

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        return self.sentences[idx], self.labels[idx], self.sen_len[idx]

class ReviewsDataSet(Dataset):
    def __init__(self, sentences, sentences_lens, y):
         self.X = sentences
         self.X_lens = sentences_lens
         self.y = y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, item):
         return self.X[item], self.X_lens[item], self.y[item]


def tokenize(x_train, x_val):
     ps = PorterStemmer()
     word2idx = {"[PAD]": 0, "[UNK]": 1}
     idx2word = ["[PAD]", "[UNK]"]
     for sent in x_train:
        for word in sent:
            if ps.stem(word) not in word2idx:
                word2idx[ps.stem(word)] = len(word2idx)
                idx2word.append(ps.stem(word))
     final_list_train, final_list_test = [], []
     for sent in x_train:
        final_list_train.append([word2idx[ps.stem(word)] for word in sent])
     for sent in x_val:
        final_list_test.append([word2idx[ps.stem(word)] if ps.stem(word) in word2idx else word2idx['[UNK]'] for word in sent])
     return final_list_train, final_list_test, word2idx, idx2word


def padding_(sentences, seq_len):
    features = np.zeros((len(sentences), seq_len), dtype=int)
    for ii, review in enumerate(sentences):
        if len(review) != 0:
            features[ii, :len(review)] = np.array(review)[:seq_len]
    return features


class LSTMNet(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim=500, hidden_dim=300, tag_dim=2):
        super(LSTMNet, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.word_embedding = nn.Embedding(vocab_size, self.embedding_dim)

        self.lstm = nn.LSTM(input_size=embedding_dim,
                                  hidden_size=hidden_dim,
                                  num_layers=3,
                                  batch_first=True,
                                  dropout=0.4,
                                  bidirectional=True
                                  )
        self.hidden2tag = nn.Sequential(nn.ReLU(),
                                        nn.Linear(self.hidden_dim * 2, self.hidden_dim),
                                        nn.Dropout(p=0.3),
                                        nn.Linear(self.hidden_dim , tag_dim),
                                        )
        self.loss_fn = FocalLoss(gamma=2., alpha=0.5)

    def forward(self, sentence, tags=None):
        embeds = self.word_embedding(sentence)
        lstm_out, _ = self.lstm(embeds)
        tag_space = self.hidden2tag(lstm_out)
        tag_score = F.softmax(tag_space, dim=1)
        if tags is None:
            return tag_score, None
        loss = self.loss_fn(tag_space, tags)
        return tag_score, loss

def fix_y(y_s, new_len):
    new_y = []
    for y in y_s:
        f = copy(y)
        gap = (new_len - len(y)) * [-1]
        f.extend(gap)
        new_y.append(f)
    return new_y

def train(x_train, y_train, x_test, y_test, epochs=100):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x_train, x_test, word2idx, idx2word = tokenize(x_train, x_test)
    train_sentence_lens = [len(s) for s in x_train]
    test_sentence_lens = [len(s) for s in x_test]
    x_train_pad = padding_(x_train, max(train_sentence_lens))
    x_test_pad = padding_(x_test, max(test_sentence_lens))
    y_train = fix_y(y_train,  max(train_sentence_lens))
    y_test = fix_y(y_test, max(test_sentence_lens))
    vocab_size = len(word2idx)
    train_dataset = ReviewsDataSet(x_train_pad, train_sentence_lens, torch.tensor(y_train))
    test_dataset = ReviewsDataSet(x_test_pad, test_sentence_lens, torch.tensor(y_test))
    train_dataloader = DataLoader(train_dataset, batch_size=64)
    test_dataloader = DataLoader(test_dataset, batch_size=64)
    model = LSTMNet(vocab_size)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=8e-3)
    best_f1 = 0
    best_epoch = None
    for epoch in range(epochs):
        print(f"\n -- Epoch {epoch} --")
        val_f1 = train_(model, device, optimizer, train_dataloader, test_dataloader)
        if val_f1 >= best_f1:
            best_accuracy = val_f1
            best_epoch = epoch
        if epoch - best_epoch == 3:
            break
    print(f"best accuracy: {best_accuracy:.2f} in epoch {best_epoch}")


def train_(model, device, optimizer, train_dataset, val_dataset):
    train_pred = []
    train_y = []
    test_pred = []
    test_y = []
    for phase in ["train", "validation"]:
        if phase == "train":
            model.train(True)
        else:
            model.train(False) #or model.evel()
        dataset = train_dataset if phase == "train" else val_dataset
        t_bar = tqdm(dataset)
        for sentences, lens, tags in t_bar:
            sentences = sentences.to(device)
            tags = tags.to(device)
            lens = lens.to(device)
            if phase == "train":
                # forward pass
                loss = 0
                for x, y, sen_len in zip(sentences, tags, lens):
                    x = x[:sen_len]
                    y = y[:sen_len]
                    tag_scores, c_loss = model(x, y)
                    loss += c_loss

                    tag_scores = tag_scores.detach().cpu().numpy()
                    pred = np.argmax(tag_scores, 1)
                    y = y.detach().cpu().numpy()
                    train_y.extend(y)
                    train_pred.extend(pred)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                training_loss = loss.detach().cpu().numpy()
            else:
                with torch.no_grad():
                    # forward pass
                    for x, y, sen_len in zip(sentences, tags, lens):
                        x = x[:sen_len]
                        y = y[:sen_len]
                        tag_scores, _ = model(x)
                        tag_scores = tag_scores.detach().cpu().numpy()
                        pred = np.argmax(tag_scores, 1)
                        y = y.detach().cpu().numpy()
                        test_y.extend(y)
                        test_pred.extend(pred)
    print(f"training F1: {f1_score(train_pred, train_y)}, loss: {training_loss}")
    print(f"validation F1: {f1_score(test_pred, test_y)}")
    return f1_score(test_pred, test_y)


