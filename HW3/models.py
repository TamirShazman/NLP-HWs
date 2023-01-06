import torch.nn as nn
from torch import cat
import torch
import math

class DependencyParser(nn.Module):
    def __init__(self, w_vocab_size, p_vocab_size, w_embedding_dim=30, p_embedding_dim=30, hidden_dim=50):
        super(DependencyParser, self).__init__()
        self.w_embedding_dim = w_embedding_dim
        self.p_embedding_dim = p_embedding_dim
        self.embedding_dim = w_embedding_dim + p_embedding_dim
        self.word_embedding = nn.Embedding(w_vocab_size, w_embedding_dim)
        self.pos_embedding = nn.Embedding(p_vocab_size, p_embedding_dim)
        self.encoder = nn.LSTM(self.embedding_dim, hidden_dim, batch_first=True, bidirectional=True, num_layers=2)
        self.edge_scorer = nn.Sequential(nn.Linear(4*hidden_dim, hidden_dim),
                                         nn.Tanh(),
                                         nn.Linear(hidden_dim, 1)
                                         )
        self.loss_function = nn.NLLLoss()
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, word_idx_tensor, pos_idx_tensor, true_tree_heads=None):
        w_embeds = self.word_embedding(word_idx_tensor)
        p_embeds = self.pos_embedding(pos_idx_tensor)
        embeds = torch.unsqueeze(cat((w_embeds, p_embeds), 1), 0)
        lstm_out, _ = self.encoder(embeds)
        lstm_out = lstm_out[0]
        # create combination vec
        X1 = lstm_out.unsqueeze(0)
        Y1 = lstm_out.unsqueeze(1)
        X2 = X1.repeat(lstm_out.shape[0], 1, 1)
        Y2 = Y1.repeat(1, lstm_out.shape[0], 1)
        Z = torch.cat([Y2, X2], -1)
        lstm_out_combi = Z.view(-1, Z.shape[-1])
        #
        score_mat_self_loop = self.edge_scorer(lstm_out_combi).view((lstm_out.shape[0], lstm_out.shape[0]))
        score_mat = score_mat_self_loop.fill_diagonal_(-math.inf)

        loss = None

        if true_tree_heads is not None:
            log_softmax_score = self.log_softmax(score_mat)
            #print("inside")
           # print(log_softmax_score[1:], true_tree_heads[1:])
            loss = self.loss_function(log_softmax_score[1:], true_tree_heads[1:]) # remove root

        pred_score_mat = score_mat.T.fill_diagonal_(0)
        pred_score_mat[:,0] = 0

        return loss, pred_score_mat
