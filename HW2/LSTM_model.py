from torch.utils.data import Dataset
import torch
import torch.nn as nn


class LSTMDataset(Dataset):
    def __init__(self, sentences, labels, sen_len):
        self.sentences = sentences
        self.labels = labels
        self.sen_len = sen_len

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        return self.sentences[idx], self.labels[idx], self.sen_len[idx]


class LSTMNet(torch.nn.Module):
    def __init__(self, input_size=200, hidden_size=100, num_layers=2, output_size=2):
        super(LSTMNet, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                                  hidden_size=hidden_size,
                                  num_layers=num_layers,
                                  batch_first=True,
                                  dropout=0.35,
                                  bidirectional=True
                                  )
        self.classifier = nn.Sequential(nn.ReLU(),
                                        nn.Linear(2*hidden_size, hidden_size),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.3),
                                        nn.Linear(hidden_size, output_size),
                                        )
        self.hidden_size = hidden_size
        self.input_size = input_size

    def forward(self, sentence):
        lstm_out, _ = self.lstm(sentence)
        return self.classifier(lstm_out)