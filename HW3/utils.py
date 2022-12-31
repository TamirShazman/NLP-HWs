import numpy as np
from torch.utils.data import Dataset
from torch.nn import functional as F
import torch

def parse_sentence(file_path):
    """
    @param file_path:
    @return: list of sentences
    """
    with open(file_path, 'r') as file:
        sentences = file.read().split('\n\n')[:-1]
    return sentences


def get_tokens(file_path):
    """
    @param file_path:
    @return: return all token in a order list
    """
    sentences = parse_sentence(file_path)
    tokens = []
    for s in sentences:
        s_t = ['root']
        for w in s.split('\n'):
            t = w.split('\t')[1]
            s_t.append(t)
        tokens.append(s_t)
    return tokens


def get_token_pos(file_path):
    """
    @param file_path:
    @return: return all token pos in a order list
    """
    sentences = parse_sentence(file_path)
    tokens_pos = []
    for s in sentences:
        s_t = ['root']
        for w in s.split('\n'):
            t = w.split('\t')[3]
            s_t.append(t)
        tokens_pos.append(s_t)
    return tokens_pos


def get_head_token(file_path):
    """
    @param file_path:
    @return: return all head token in a order list
    """
    sentences = parse_sentence(file_path)
    head_token = []
    for s in sentences:
        s_t = [-1]
        for w in s.split('\n'):
            t = w.split('\t')[6]
            s_t.append(int(t)) # -1 for indexing
        head_token.append(s_t)
    return head_token


def convert_to_tokenized(sentences, word2idx):
    """
    @param sentences:
    @param word2idx:
    @return:
    """
    tokens = []
    for s in sentences:
        t = []
        for word in s:
            w = word.lower()
            if w in word2idx.keys():
                t.append(word2idx[w])
            else:
                t.append(word2idx["[UNK]"])
        tokens.append(t)
    return tokens

def tokenize(x_train):
    """
    @param x_train:
    @return: token dict
    """
    word2idx = {"[PAD]": 0, "[UNK]": 1}
    idx2word = ["[PAD]", "[UNK]"]
    for sent in x_train:
        for word in sent:
            w = word.lower()
            if w not in word2idx:
                word2idx[w] = len(word2idx)
                idx2word.append(w)

    return word2idx, idx2word


def padding_(sentences, seq_len):
    """
    @param sentences:
    @param seq_len:
    @return:
    """
    features = np.zeros((len(sentences), seq_len), dtype=int)
    for ii, s in enumerate(sentences):
        if len(s) != 0:
            features[ii, :len(s)] = np.array(s)[:seq_len]
    return features


def calculate_UAS(pred_tree, gt_tree):
    assert len(pred_tree) == len(gt_tree), "Predicted tree has to be in the same length of the ground truth tree"
    total = len(pred_tree)
    correct = sum([1 for i, val in enumerate(pred_tree) if (val) == gt_tree[i]])
    return correct / total

class ParserDataSet(Dataset):
    def __init__(self, sentences, pos, sentences_lens, true_trees=None):
        self.sentences = sentences
        self.pos = pos
        self.s_lens = sentences_lens
        self.true_tree = true_trees

    def __len__(self):
        return self.sentences.shape[0]

    def __getitem__(self, item):
        return self.sentences[item], self.pos[item], self.s_lens[item],self.true_tree[item] if self.true_tree is not None else None
