from gensim.models import KeyedVectors
import torch
import copy
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score

from preprocessing import get_label, download_model, break_file_to_sentences, find_word_rep
from LSTM_model import LSTMDataset
from moduls import lstm_prediction

def stack_sen_label(model, sentences, labels):
    final_x = []
    final_y = []
    sentences_len = [len(s) for s in sentences]
    counter = 0
    sentence_mean_len = max(sentences_len) + 1
    for s in sentences:
        x = []
        y = []
        gap = sentence_mean_len - len(s)
        pad = ['pad'] * gap
        new_s = copy.copy(s)
        new_s.extend(pad)
        for w in new_s:
            word_rep = find_word_rep(w, model)
            if word_rep is None:
                x.append(model['nonsense'])
            else:
                x.append(word_rep)
            if w != 'pad':
                y.append(labels[counter])
                counter += 1
            else:
                y.append(0)
        final_x.append(x)
        final_y.append(y)
    return torch.tensor(final_x), torch.tensor(final_y), torch.tensor(sentences_len)

def main():
    train_path = 'data/train.tagged'
    test_path = 'data/dev.tagged'
    path_to_word_rep = 'word_rep/'

    model_path = download_model('word2vec', path_to_word_rep)
    model = KeyedVectors.load(model_path)

    # create training dataset
    sentences = break_file_to_sentences(train_path)
    y = get_label(train_path)
    X, y, sen_len = stack_sen_label(model, sentences, y)
    train_dataset = LSTMDataset(X, y, sen_len)

    # create training dataset
    sentences = break_file_to_sentences(test_path)
    y_test = get_label(test_path)
    X, y, sen_len = stack_sen_label(model, sentences, y_test)
    test_dataset = LSTMDataset(X, y, sen_len)

    # kf = KFold(n_splits=2, random_state=None, shuffle=False)
    # for train_index, test_index in kf.split(X):
    #     X_train, X_test = X[train_index], X[test_index]
    #     Y_train, Y_test = y[train_index], y[test_index]
    #     sen_len_train, sen_len_test = sen_len[train_index], sen_len[test_index]
    #     train_dataset = LSTMDataset(X_train, Y_train, sen_len_train)
    #     test_dataset = LSTMDataset(X_test, Y_test, sen_len_test)
    pred = lstm_prediction(train_dataset, test_dataset, 300)
    print(f"The F1: score is :{f1_score(pred, y_test)}")


if __name__ == '__main__':
    main()
