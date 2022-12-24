from gensim.models import KeyedVectors
import numpy as np
from sklearn.metrics import f1_score
import torch
import copy
from LSTM_model import LSTMDataset
from moduls import lstm_prediction

from preprocessing import get_label, download_model, break_file_to_sentences, convert_sentence_presentation_to_mean, \
    find_word_rep
from moduls import knn_prediction, dl_prediction

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
    training_matrix = []
    for sentence in break_file_to_sentences(train_path):
        training_matrix.append(convert_sentence_presentation_to_mean(sentence, model, use_pos_embeded=False))
        # training_matrix.append(convert_sentence_presentation_to_concatenation(sentence, model))
    training_matrix = np.concatenate(training_matrix)
    training_labels = np.array(get_label(train_path))
    assert training_matrix.shape[0] == training_labels.shape[0], "Number of sample and labels is not the same"

    test_matrix = []
    for sentence in break_file_to_sentences(test_path):
        test_matrix.append(convert_sentence_presentation_to_mean(sentence, model, use_pos_embeded=False))
        # test_matrix.append(convert_sentence_presentation_to_concatenation(sentence, model))
    test_matrix = np.concatenate(test_matrix)
    test_labels = np.array(get_label(test_path))
    assert test_matrix.shape[0] == test_labels.shape[0], "Number of sample and labels is not the same"
    print('input size:')
    print(training_matrix.shape[1])
    # predict knn
    pred = knn_prediction(training_matrix, training_labels, test_matrix)
    print('knn prediction')
    print(f"The F1: score is :{f1_score(pred, test_labels)}")
    # predict FC
    print('deep learning basic prediction')
    pred = dl_prediction(training_matrix, training_labels, test_matrix, test_labels, input_size=training_matrix.shape[1])
    assert len(pred) == len(test_labels), "prediction and test not same length"
    print(f"The F1: score is :{f1_score(pred, test_labels)}")
    # predict LSTM
    model_path = download_model('word2vec', path_to_word_rep)
    model = KeyedVectors.load(model_path)
    X, y, sen_len = stack_sen_label(model, break_file_to_sentences(train_path), get_label(train_path))
    train_dataset = LSTMDataset(X, y, sen_len)
    X, y, sen_len = stack_sen_label(model, break_file_to_sentences(test_path), get_label(test_path))
    test_dataset = LSTMDataset(X, y, sen_len)
    pred = lstm_prediction(train_dataset, test_dataset, 300)
    print(f"The F1: score is :{f1_score(pred, get_label(test_path))}")

if __name__ == '__main__':
    main()
