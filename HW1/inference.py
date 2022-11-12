from preprocessing import read_test
from tqdm import tqdm
import numpy as np

from preprocessing import represent_input_with_features


def memm_viterbi(sentence, pre_trained_weights, feature2id, beam=3):
    """
    Write your MEMM Viterbi implementation below
    You can implement Beam Search to improve runtime
    Implement q efficiently (refer to conditional probability definition in MEMM slides)
    """
    pred = []
    all_tags = list(feature2id.feature_statistics.tags)

    # assumed the probably in pi_dict is not normalized (don't need to because the arg-max is wanted).
    # (u, v) : (not-normalized probability, back-pointer).
    pi_dict_arr = [{(u, v): (1, None) for u in all_tags for v in all_tags}]

    for i in range(2, len(sentence) - 1):

        # array that contain the pi_matrix as follow (u, v, max probability, max arg)
        curr_pi_list = []

        for (u, v) in zip(all_tags, all_tags):

            all_t_values = []

            for t in all_tags:

                # create history
                history = (sentence[i], v, sentence[i - 1], u, sentence[i - 2], t, sentence[i + 1])

                normalized_value = find_normalized_value(history, all_tags, pre_trained_weights, feature2id)

                # beam search
                if (t, v) not in pi_dict_arr[i - 2].keys():
                    continue

                # calculate not normalized probability
                prob = pred_hist(history, pre_trained_weights, feature2id) / normalized_value

                # insert value pi_value * probability
                all_t_values.append((pi_dict_arr[i - 2][t, v][0] * prob, t))

            # find max, the length can be zero because of the beam search
            if len(all_t_values) > 0:
                all_t_values.sort(key=lambda tup: tup[0], reverse=True)
                max_p, max_t = all_t_values[0]

                curr_pi_list.append((u, v, max_p, max_t))

        # beam search
        if beam is not None:
            curr_pi_list.sort(key=lambda tup: tup[2], reverse=True)
            curr_pi_list = curr_pi_list[: beam]

        # convert to dict
        curr_pi_dict = {(u, v): (max_p, max_t) for (u, v, max_p, max_t) in curr_pi_list}
        pi_dict_arr.append(curr_pi_dict)

    last_pi_list = [(u, v, p) for (u, v), (p, _) in pi_dict_arr[-1].items()]
    last_pi_list.sort(key=lambda tup: tup[2], reverse=True)
    t_1, t_2 = last_pi_list[0][0:2]
    pred.append(t_1), pred.append(t_2)
    for i in reversed(range(2, len(sentence) - 2)):
        cur_t = pi_dict_arr[i][t_1, t_2][1]
        pred.append(cur_t)
        t_1, t_2 = cur_t, t_1

    return pred


def pred_hist(history, weights, feature2id):
    """
    :param history: {c_word, c_tag, p_word, p_tag, pp_word, pp_tag, n_word}
    :param weights: pre-trained weights
    :param feature2id:
    :return: unnormalized prediction
    """
    features = represent_input_with_features(history, feature2id.feature_to_idx)
    features_vec = np.zeros(len(weights))
    features_vec[features] = 1
    return np.exp(features_vec @ weights)


def find_normalized_value(history, tags, weights, feature2id):
    """
    :param feature2id:
    :param history:
    :param tags:
    :param weights:
    :return:
    """
    exp_sum = 0
    for v in tags:
        history = (history[0], v, history[2], history[3], history[4], history[5], history[6])
        exp_sum += pred_hist(history, weights, feature2id)

    return exp_sum


def tag_all_test(test_path, pre_trained_weights, feature2id, predictions_path):
    tagged = "test" in test_path
    test = read_test(test_path, tagged=tagged)

    output_file = open(predictions_path, "a+")

    for k, sen in tqdm(enumerate(test), total=len(test)):
        sentence = sen[0]
        pred = memm_viterbi(sentence, pre_trained_weights, feature2id)[1:]
        sentence = sentence[2:]
        for i in range(len(pred)):
            if i > 0:
                output_file.write(" ")
            output_file.write(f"{sentence[i]}_{pred[i]}")
        output_file.write("\n")
    output_file.close()
