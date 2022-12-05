from gensim.models import KeyedVectors
import torch
import copy
from sklearn.model_selection import KFold

from preprocessing import get_label, download_model, break_file_to_sentences, find_word_rep
from LSTM_model import ReviewsDataSet, train
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
                y.append(-1)
        final_x.append(x)
        final_y.append(y)
    return torch.tensor(final_x), torch.tensor(final_y), torch.tensor(sentences_len)

def main():
    train_path = 'data/train_new.tagged'
    test_path = 'data/test_real.tagged'
    path_to_word_rep = 'word_rep/'

    # model_path = download_model('glove', path_to_word_rep)
    # model = KeyedVectors.load(model_path)

    # create training dataset
    train_sentences = break_file_to_sentences(train_path)
    sen_len = [len(s) for s in train_sentences]
    labels = get_label(train_path)
    train_y_s = []
    counter = 0
    for s in train_sentences:
        y = []
        for i in range(len(s)):
            y.append(labels[counter])
            counter += 1
        train_y_s.append(y)

    # create test dataset
    test_sentences = break_file_to_sentences(test_path)
    sen_len = [len(s) for s in test_sentences]
    labels = get_label(test_path)
    test_y_s = []
    counter = 0
    for s in test_sentences:
        y = []
        for i in range(len(s)):
            y.append(labels[counter])
            counter += 1
        test_y_s.append(y)

    train(train_sentences, train_y_s, test_sentences, test_y_s)
    # kf = KFold(n_splits=2, random_state=None, shuffle=False)
    # for train_index, test_index in kf.split(X):
    #     X_train, X_test = X[train_index], X[test_index]
    #     Y_train, Y_test = y[train_index], y[test_index]
    #     sen_len_train, sen_len_test = sen_len[train_index], sen_len[test_index]
    #     train_dataset = LSTMDataset(X_train, Y_train, sen_len_train)
    #     test_dataset = LSTMDataset(X_test, Y_test, sen_len_test)
    lstm_prediction(train_dataset, test_dataset, 200)


if __name__ == '__main__':
    main()
