from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay,accuracy_score, recall_score, precision_score, f1_score
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

import pickle
from preprocessing import preprocess_train
from optimization import get_optimal_vector
from inference import tag_all_test

def get_sentences_labels(file_path):
    labels = list()
    features = list()
    # extract test ground truth
    with open(file_path) as file:
        for line in file:
            sentence = list()
            sentence_POS = list()
            if line[-1:] == "\n":
                line = line[:-1]
            split_words = line.split(' ')

            for word_idx in range(len(split_words)):
                cur_word, cur_tag = split_words[word_idx].split('_')
                sentence.append(cur_word)
                sentence_POS.append(cur_tag)
            features.append(sentence)
            labels.append(sentence_POS)

    return features, labels

def get_ground_and_predicted(test_file, prediction_file):
    test_labels = list()
    predicted_labels = list()
    # extract test ground truth
    with open(test_file) as file:
        for line in file:
            if line[-1:] == "\n":
                line = line[:-1]
            split_words = line.split(' ')

            for word_idx in range(len(split_words)):
                _, cur_tag = split_words[word_idx].split('_')
                test_labels.append(cur_tag)

    # extract the predicted labels
    i=1
    with open(prediction_file) as file:
        for line in file:
            if line[-1:] == "\n":
                line = line[:-1]
            split_words = line.split(' ')
            try:
                for word_idx in range(len(split_words)):
                    _, cur_tag = split_words[word_idx].split('_')
                    predicted_labels.append(cur_tag)
            except:
                print(i, line)
            i = i + 1
    # TODO: REMOVE BEFORE SUBMITTING
    min_labels = min(len(test_labels), len(predicted_labels))
    test_labels = test_labels[:min_labels]
    predicted_labels = predicted_labels[:min_labels]
    return test_labels, predicted_labels,


def get_accuracy(ground_truth, predicted):
    return accuracy_score(ground_truth, predicted)

def show_confusion_matrix(ground_truth, predicted, grading_metric):
    labels = list(set(ground_truth))
    scores = grading_metric(ground_truth, predicted, average=None, labels=labels)
    top_ten_indices = sorted(range(len(scores)), key=lambda i: scores[i])[-10:]
    top_ten_labels = [labels[i] for i in top_ten_indices]
    cm = confusion_matrix(ground_truth, predicted, labels=top_ten_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels = top_ten_labels)
    disp.plot()
    plt.show()

def test():
    ground_truth, predicted = get_ground_and_predicted("C:\\Users\\dovid\\PycharmProjects\\NLP\\NLP-HWs\\HW1\\data\\test1.wtag",
                                                       'C:\\Users\\dovid\\PycharmProjects\\NLP\\NLP-HWs\\HW1\\predictions.wtag')
    print(get_accuracy(ground_truth, predicted))
    show_confusion_matrix(ground_truth, predicted, precision_score)

def cross_validation(file_path):
    threshold = 1
    lam = 1
    fold_train_path = 'fold_train.wtag'
    fold_weight_path = 'fold_weight.wtag'
    fold_test_path = 'fold_test.wtag'
    fold_prediction_path = 'fold_prediction.wtag'

    dataset = list()
    with open(file_path) as file:
        for line in file:
            if line[-1:] == "\n":
                line = line[:-1]
                dataset.append(line)

    # sentences, sentence_labels = get_sentences_labels(file_path)
    kf = KFold(n_splits=4, random_state=None, shuffle=True)
    cv_results = list()
    for train_index, test_index in kf.split(dataset):
        train_dataset = [dataset[i] for i in train_index]
        test_dataset = [dataset[i] for i in test_index]
        with open(fold_train_path, mode='wt') as myfile:
            myfile.write('\n'.join(train_dataset))
        with open(fold_test_path, mode='wt') as myfile:
            myfile.write('\n'.join(test_dataset))

        statistics, feature2id = preprocess_train(fold_train_path, threshold)
        get_optimal_vector(statistics=statistics, feature2id=feature2id, weights_path=fold_weight_path, lam=lam)

        with open(fold_weight_path, 'rb') as f:
            optimal_params, feature2id = pickle.load(f)

        tag_all_test(fold_test_path, pre_trained_weights, feature2id, fold_prediction_path)
        pre_trained_weights = optimal_params[0]

        ground_truth, predicted = get_ground_and_predicted(fold_test_path, fold_prediction_path)

        cv_results.append(get_accuracy(ground_truth, predicted))

    print(cv_results)
    cv_avg = sum(cv_results / len(cv_results)
    print(cv_avg)

if __name__ == '__main__':
    test()





