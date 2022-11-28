from gensim.models import KeyedVectors
import numpy as np
from sklearn.metrics import f1_score

from preprocessing import get_label, download_model, break_file_to_sentences, convert_sentence_presentation_to_mean, convert_sentence_presentation_to_concatenation
from moduls import knn_prediction, dl_prediction


def main():
    train_path = 'data/train.tagged'
    test_path = 'data/dev.tagged'
    path_to_word_rep = 'word_rep/'

    model_path = download_model('glove', path_to_word_rep)
    model = KeyedVectors.load(model_path)

    # create training dataset
    training_matrix = []
    for sentence in break_file_to_sentences(train_path):
        # training_matrix.append(convert_sentence_presentation_to_mean(sentence, model, use_pos_embeded=False))
        training_matrix.append(convert_sentence_presentation_to_concatenation(sentence, model))
    training_matrix = np.concatenate(training_matrix)
    training_labels = np.array(get_label(train_path))
    assert training_matrix.shape[0] == training_labels.shape[0], "Number of sample and labels is not the same"

    test_matrix = []

    for sentence in break_file_to_sentences(test_path):
        # test_matrix.append(convert_sentence_presentation_to_mean(sentence, model, use_pos_embeded=False))
        test_matrix.append(convert_sentence_presentation_to_concatenation(sentence, model))
    test_matrix = np.concatenate(test_matrix)
    test_labels = np.array(get_label(test_path))
    assert test_matrix.shape[0] == test_labels.shape[0], "Number of sample and labels is not the same"
    print('input size:')
    print(training_matrix.shape[1])
    # predict
    # pred = knn_prediction(training_matrix, training_labels, test_matrix)
    # print('knn prediction')
    # print(f"The F1: score is :{f1_score(pred, test_labels)}")

    print('deep learning basic prediction')
    pred = dl_prediction(training_matrix, training_labels, test_matrix, test_labels, input_size=training_matrix.shape[1])
    assert len(pred) == len(test_labels), "prediction and test not same length"
    print(f"The F1: score is :{f1_score(pred, test_labels)}")


if __name__ == '__main__':
    main()
