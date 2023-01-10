import torch.nn as nn
from torch import cat
import torch
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class DependencyParser(nn.Module):
    def __init__(self, w_vocab_size, p_vocab_size, w_embedding_dim=30, p_embedding_dim=30, hidden_dim=50, init_emb_path='embedding/mtembedding.pt'):
        super(DependencyParser, self).__init__()
        self.w_embedding_dim = w_embedding_dim
        self.p_embedding_dim = p_embedding_dim
        self.embedding_dim = w_embedding_dim + p_embedding_dim
        self.word_embedding = nn.Embedding(w_vocab_size, w_embedding_dim)
        if init_emb_path:
            emb = torch.load(init_emb_path)
            with torch.no_grad():
                self.word_embedding.weight = emb
            self.word_embedding.requires_grad_(False)
        self.pos_embedding = nn.Embedding(p_vocab_size, p_embedding_dim)
        self.encoder = nn.LSTM(self.embedding_dim, hidden_dim, batch_first=True, bidirectional=True, num_layers=2, dropout=0.15)
        self.edge_scorer = nn.Sequential(nn.Linear(4*hidden_dim, hidden_dim),
                                         nn.Tanh(),
                                         nn.Dropout(0.15),
                                         nn.Linear(hidden_dim, 1)
                                         )
        self.loss_function = nn.NLLLoss()
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, word_idx_tensor, pos_idx_tensor, true_tree_heads=None):
        with torch.no_grad():
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


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)



class CBOW_Model(nn.Module):
    def __init__(self, vocab_size):
        super(CBOW_Model, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=300, max_norm=1)
        self.linear = nn.Linear(in_features=300, out_features=vocab_size)

    def forward(self, inputs_):
        x = self.embedding(inputs_)
        x = x.mean(axis=1)
        x = self.linear(x)
        return x

class CombinationDecoder(nn.Module):
    def __init__(self, in_dim):
        super(CombinationDecoder, self).__init__()
        mid = int(in_dim / 2)
        self.edge_scorer = nn.Sequential(nn.Linear(in_dim, mid),
                                         nn.Tanh(),
                                         nn.Linear(mid, 1)
                                         )
        self.loss_function = nn.NLLLoss()
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, src, true_tree_heads=None):
        # create combination vec
        src = src[0]
        X1 = src.unsqueeze(0)
        Y1 = src.unsqueeze(1)
        X2 = X1.repeat(src.shape[0], 1, 1)
        Y2 = Y1.repeat(1, src.shape[0], 1)
        Z = torch.cat([Y2, X2], -1)
        lstm_out_combi = Z.view(-1, Z.shape[-1])
        #
        score_mat_self_loop = self.edge_scorer(lstm_out_combi).view((src.shape[0], src.shape[0]))
        score_mat = score_mat_self_loop.fill_diagonal_(-math.inf)

        loss = None

        if true_tree_heads is not None:
            log_softmax_score = self.log_softmax(score_mat)
            # print("inside")
            # print(log_softmax_score[1:], true_tree_heads[1:])
            loss = self.loss_function(log_softmax_score[1:], true_tree_heads[1:])  # remove root

        pred_score_mat = score_mat.T.fill_diagonal_(0)
        pred_score_mat[:, 0] = 0

        return loss, pred_score_mat

class TransformerModel(nn.Module):

    def __init__(self, w_vocab_size, p_vocab_size, w_emb_size, pos_emb_size, nhead, dim_feedforward, nlayers, dropout=0.5, task_type='MMP' , init_emb_path='embedding/mtembedding.pt'):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer ' + task_type
        self.w_emb = nn.Embedding(w_vocab_size, w_emb_size)
        if init_emb_path:
            emb = torch.load(init_emb_path)
            with torch.no_grad():
                self.w_emb.weight = emb
            self.w_emb.requires_grad_(False)
        self.p_emb = nn.Embedding(p_vocab_size, pos_emb_size)
        self.embedding_size = w_emb_size + pos_emb_size
        self.pos_encoder = PositionalEncoding(self.embedding_size, dropout)
        encoder_layers = TransformerEncoderLayer(self.embedding_size, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        if task_type == 'MMP':
            self.decoder = nn.Linear(self.embedding_size, w_vocab_size)
        else:
            self.decoder = CombinationDecoder(2*self.embedding_size)

        #self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.p_emb.weight.data.uniform_(-initrange, initrange)
        self.w_emb.weight.data.uniform_(-initrange, initrange)
        if self.model_type == 'TransformerMMP':
            self.decoder.bias.data.zero_()
            self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, sens, poss, src_mask=None, true_tree_heads=None):
        emb = cat((self.w_emb(sens) ,self.p_emb(poss)),1)* math.sqrt(self.embedding_size)
        if self.model_type == 'TransformerMMP ':
            emb = self.pos_encoder(emb)
            output = self.transformer_encoder(emb, src_mask)
            output = self.decoder(output)
            return output
        else:
            emb = torch.unsqueeze(emb, 0)
            emb = self.pos_encoder(emb)
            output = self.transformer_encoder(emb)
            loss, pred_score_mat = self.decoder(output, true_tree_heads)
            return loss, pred_score_mat