from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay,accuracy_score, recall_score, precision_score, f1_score
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, train_test_split
import os
import shutil
import numpy as np
from datetime import datetime
import time

import pickle
from preprocessing import preprocess_train
from optimization import get_optimal_vector
from inference import tag_all_test


def get_sentences_labels(file_path):
    labels = list()
    sentences = list()
    # extract test ground truth
    with open(file_path) as file:
        for line in file:
            sentence = list()
            sentence_POS = list()
            if line[-1:] == "\n":
                line = line[:-1]
            split_words = line.split(' ')

            for word_idx in range(len(split_words)):
                try:
                    cur_word, cur_tag = split_words[word_idx].split('_')
                    sentence.append(cur_word)
                    sentence_POS.append(cur_tag)
                except Exception as e:
                    print(e)
                    print(word_idx)
                    print(line)
            sentences.append(sentence)
            labels.append(sentence_POS)

    return sentences, labels


def get_ground_and_predicted(test_file, prediction_file):
    """

    @param test_file:
    @param prediction_file:
    @return:
    """
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
            # TODO: Remove before submitting
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
    return test_labels, predicted_labels


def get_accuracy(ground_truth, predicted):
    return accuracy_score(ground_truth, predicted)


def show_confusion_matrix(ground_truth, predicted, grading_metric):
    labels = list(set(ground_truth))
    scores = grading_metric(ground_truth, predicted, average=None, labels=labels)
    top_ten_indices = sorted(range(len(scores)), key=lambda i: scores[i])[:10]
    top_ten_labels = [labels[i] for i in top_ten_indices]
    cm = confusion_matrix(ground_truth, predicted, labels=top_ten_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels = top_ten_labels)
    disp.plot()
    plt.show()

def test(ground_truth_path, predicted_path):
    ground_truth, predicted = get_ground_and_predicted(ground_truth_path, predicted_path)
    print(get_accuracy(ground_truth, predicted))
    show_confusion_matrix(ground_truth, predicted, f1_score)

def cross_validation(file_path, threshold, lam, splits, iteration=0):
    """
    Performs cross validation on a given data set
    @param file_path:
    @param return_all:
    @return:
    """
    isExist = os.path.exists('temp')
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs('temp')
    else:
        shutil.rmtree('temp')
        os.makedirs('temp')
    # threshold = 10
    # lam = 1
    # splits = 2
    fold_train_path = 'temp/fold_train.wtag'
    fold_weight_path = 'temp/fold_weight.pkl'
    fold_test_path = 'temp/fold_test.wtag'
    fold_prediction_path = 'temp/fold_prediction.wtag'

    dataset = list()
    with open(file_path) as file:
        for line in file:
            if line[-1:] == "\n":
                line = line[:-1]
                dataset.append(line)

    # sentences, sentence_labels = get_sentences_labels(file_path)
    kf = KFold(n_splits=splits, random_state=None, shuffle=True)
    cv_results = list()

    for iter, (train_index, test_index) in enumerate(kf.split(dataset)):
        if iter > 2:
            break
        else:
            train_dataset = [dataset[i] for i in train_index]
            test_dataset = [dataset[i] for i in test_index]
            with open(fold_train_path, mode='wt') as myfile:
                myfile.write('\n'.join(train_dataset))
            with open(fold_test_path, mode='wt') as myfile:
                myfile.write('\n'.join(test_dataset))

            statistics, feature2id = preprocess_train(fold_train_path, threshold)
            numbered_fold_weight_path = fold_weight_path[:-4] + str(iter) + str(iteration)+ fold_weight_path[-4:]
            get_optimal_vector(statistics=statistics, feature2id=feature2id, weights_path=numbered_fold_weight_path, lam=lam)

            with open(numbered_fold_weight_path, 'rb') as f:
                optimal_params, feature2id = pickle.load(f)

            pre_trained_weights = optimal_params[0]
            numbered_fold_prediction_path = fold_prediction_path[:-5] + str(iter) + str(iteration) + fold_prediction_path[-5:]
            tag_all_test(fold_test_path, pre_trained_weights, feature2id, numbered_fold_prediction_path)

            ground_truth, predicted = get_ground_and_predicted(fold_test_path, numbered_fold_prediction_path)

            curr_accuracy = get_accuracy(ground_truth, predicted)
            cv_results.append(curr_accuracy)

    print("Threshold", threshold)
    print("Lam", lam)
    print("CV results:")
    print(cv_results)
    max_value = max(cv_results)
    index = cv_results.index(max_value)
    print("Best performance by fold {} with {} accuracy".format(index, max_value))
    cv_avg = sum(cv_results) / len(cv_results)
    print("Avg. of CV results", cv_avg)
    print()

    return fold_weight_path[:-4] + str(index) + str(iteration) + fold_weight_path[-4:]

def train(file_path, threshold, lam, iteration=0):
    """
    Trains on whole (test) set
    @param file_path:
    @param return_all:
    @return:
    """
    isExist = os.path.exists('temp')
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs('temp')
    else:
        shutil.rmtree('temp')
        os.makedirs('temp')
    # threshold = 10
    # lam = 1
    # splits = 2
    fold_train_path = 'temp/fold_train.wtag'
    fold_weight_path = 'temp/fold_weight.pkl'
    fold_prediction_path = 'temp/fold_prediction.wtag'

    # dataset = list()
    # with open(file_path) as file:
    #     for line in file:
    #         if line[-1:] == "\n":
    #             line = line[:-1]
    #             dataset.append(line)

    statistics, feature2id = preprocess_train(file_path, threshold)
    get_optimal_vector(statistics=statistics, feature2id=feature2id, weights_path=fold_weight_path, lam=lam)

    return fold_weight_path


def add_weak_labels(unlabeled_data_path, predicted_path, weak_label_path, to_label_indices):
    # modify unlabeled data that should be inferred next round
    unlabeled_dataset = list()
    with open(unlabeled_data_path) as file:
        for index, line in enumerate(file):
            if line[-1:] == "\n":
                line = line[:-1]
                if index not in to_label_indices:
                    unlabeled_dataset.append(line)
    with open(unlabeled_data_path, mode='wt') as myfile:
        myfile.write('\n'.join(unlabeled_dataset))

    # add weak labels
    weak_labeled_dataset = list()
    with open(weak_label_path) as file:
        for index, line in enumerate(file):
            if line[-1:] == "\n":
                line = line[:-1]
                weak_labeled_dataset.append(line)
    with open(predicted_path) as file:
        for index, line in enumerate(file):
            if line[-1:] == "\n":
                line = line[:-1]
                if index in to_label_indices:
                    weak_labeled_dataset.append(line)
    with open(weak_label_path, mode='wt') as myfile:
        myfile.write('\n'.join(weak_labeled_dataset))


def ssl(labeled_path, unlabeled_path, iter, probability_threshold, feature_threshold, lam, splits, final=False):
    isExist = os.path.exists('ssl')
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs('ssl')
    else:
        shutil.rmtree('ssl')
        os.makedirs('ssl')
    # iter = 5
    # threshold = 0.2

    weak_label_path = 'ssl/weak_label.wtag'
    test_labeled_path = 'ssl/test_labeled.wtag'
    to_be_predicted_path = 'ssl/to_be_predicted.words'
    predictions_path = 'ssl/prediction.wtag'
    final_prediction_path = 'temp/final_test.wtag'


    # set aside objective truth and training set
    dataset = list()
    with open(labeled_path) as file:
        for line in file:
            if line[-1:] == "\n":
                line = line[:-1]
                dataset.append(line)
    train_dataset, test_dataset = train_test_split(dataset)
    # train_dataset = [dataset[i] for i in train_index]
    # test_dataset = [dataset[i] for i in test_index]

    with open(test_labeled_path, mode='wt') as myfile:
        myfile.write('\n'.join(test_dataset))
    if final != True:
        with open(weak_label_path, mode='wt') as myfile:
            myfile.write('\n'.join(train_dataset))
    else:
        shutil.copy2(labeled_path, weak_label_path)
    # set aside labels to be predicted
    shutil.copy2(unlabeled_path, to_be_predicted_path)

    # each iteratin get most confident predictions and use as weak labels
    for i in range(iter):
        probability_scores = list()
        if final != True:
            current_weights_path = cross_validation(weak_label_path, feature_threshold, lam, splits, iteration=iter)
        else:
            current_weights_path = train(weak_label_path, feature_threshold, lam, iter)
        if i == iter - 1: # no point in predicting weak labels
            break
        else:
            with open(current_weights_path, 'rb') as f:
                optimal_params, feature2id = pickle.load(f)
            numbered_prediction_path = predictions_path[:-5] + str(iter) + str(
                i) + predictions_path[-5:]
            tag_all_test(to_be_predicted_path, optimal_params[0], feature2id, numbered_prediction_path, probability_scores)

            top_indices = [index for index, score in zip(range(len(probability_scores)), probability_scores) if score >= probability_threshold]
            if len(top_indices) < 5:
                break
            else:
                print("Adding {} weak labels".format(len(top_indices)))
                add_weak_labels(to_be_predicted_path, numbered_prediction_path, weak_label_path, top_indices)
            # evaluate ssl model
            with open(current_weights_path, 'rb') as f:
                optimal_params, feature2id = pickle.load(f)
            tag_all_test(test_labeled_path, optimal_params[0], feature2id, numbered_prediction_path[:-6] + "test" + numbered_prediction_path[-6:])
            print(i, "RESULTS ON TEST SET:")
            test(test_labeled_path, numbered_prediction_path[:-6] + "test" + numbered_prediction_path[-6:])
    # evaluate ssl model
    print("Final weights path", current_weights_path)
    with open(current_weights_path, 'rb') as f:
        optimal_params, feature2id = pickle.load(f)
    tag_all_test(test_labeled_path, optimal_params[0], feature2id, numbered_prediction_path[:-6] + "FINAL" + numbered_prediction_path[-6:])
    print("FINAL RESULTS ON TEST SET:")
    test(test_labeled_path, numbered_prediction_path[:-6] + "FINAL" + numbered_prediction_path[-6:])




if __name__ == '__main__':
    # get_sentences_labels('ssl/weak_label.wtag')
    start_time = datetime.now()
    ssl('data/train2.wtag', 'data/comp2.words', 3, 0.4, 1, 0.5, 4, final=True)
    print((datetime.now() - start_time).total_seconds(), "seconds")
