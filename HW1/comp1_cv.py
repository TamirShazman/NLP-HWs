import os
import numpy as np
from sklearn.model_selection import KFold
import pickle

from preprocessing import preprocess_train
from optimization import get_optimal_vector
from inference import tag_all_test
from evaluation import get_ground_and_predicted, get_accuracy

"""
This
"""

MAX_Threshold = 300
MIN_Threshold = 25
Threshold_NUMBERS = 3
MAX_lambda = 3
MIN_lambda = 0.5
lambda_NUMBERS = 3
CV_SPLITS = 3
RANDOM_SEED = 42


def file_to_list(path):
    """
    :param path: file path
    :return: list of sentences
    """
    list_of_sentences = []
    with open(path) as f:
        for line in f:
            list_of_sentences.append(line)
    return list_of_sentences


def list_to_file(path, s_list):
    """
    :param s_list: list of sentences
    :param path: path to save the list
    :return: create file with the list
    """
    assert not os.path.isfile(path), f"The file named {path}, is already exist"
    output_file = open(path, "w+")
    for line in s_list:
        if line[-1] == '\n':
            line = line[:-1]
        output_file.write(line)
        output_file.write('\n')
    output_file.close()


def cross_val(data, tmp_folder, threshold, lam):
    """
    :param lam:
    :param threshold:
    :param data:
    :param tmp_folder:
    :return:
    """
    train_path = tmp_folder + 'train.wtag'
    test_path = tmp_folder + 'test.wtag'
    prediction_path = tmp_folder + 'pred.wtag'
    weight_path = tmp_folder + 'weight.pkl'
    kf = KFold(n_splits=CV_SPLITS, random_state=RANDOM_SEED, shuffle=True)

    acc_list = []
    for i, (train_index, test_index) in enumerate(kf.split(data)):
        train_dataset = [data[i] for i in train_index]
        test_dataset = [data[i] for i in test_index]
        list_to_file(path=train_path, s_list=train_dataset)
        list_to_file(path=test_path, s_list=test_dataset)
        statistics, feature2id = preprocess_train(train_path, threshold)
        get_optimal_vector(statistics=statistics, feature2id=feature2id, weights_path=weight_path, lam=lam)

        with open(weight_path, 'rb') as f:
            optimal_params, feature2id = pickle.load(f)

        pre_trained_weights = optimal_params[0]
        tag_all_test(test_path, pre_trained_weights, feature2id, prediction_path)
        ground_truth, predicted = get_ground_and_predicted(test_path, prediction_path)
        curr_accuracy = get_accuracy(ground_truth, predicted)
        print(f"accuracy for fold {i}, is {curr_accuracy}")

        acc_list.append(curr_accuracy)

        os.remove(train_path), os.remove(test_path), os.remove(prediction_path), os.remove(weight_path)
    return sum(acc_list) / len(acc_list)


def main():
    cv_result_path = 'cv_results.txt'
    train_path = "data/train1.wtag"
    test_path = "data/test1.wtag"
    tmp_folder = "tmp/"

    if not os.path.isfile(tmp_folder):
        os.mkdir(tmp_folder)

    dataset = []
    for path in [train_path, test_path]:
        dataset.extend(file_to_list(path))

    thresholds_list = np.linspace(MIN_Threshold, MAX_Threshold, Threshold_NUMBERS).tolist()
    lambda_list = np.linspace(MIN_lambda, MAX_lambda, lambda_NUMBERS).tolist()

    for threshold in thresholds_list:
        for lam in lambda_list:
            cv_acc = cross_val(dataset, tmp_folder, threshold, lam)

            with open(cv_result_path, 'a') as f:
                f.write(f'Threshold: {threshold}, lambda: {lam}, cv accuracy: {cv_acc} \n')

            exit()


if __name__ == '__main__':
    main()
