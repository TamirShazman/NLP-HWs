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

        u_v_val_dict = {}
        u_t_val_dict = {}

        for v in all_tags:

            for u in all_tags:

                for t in all_tags:

                    # beam search
                    if (t, u) not in pi_dict_arr[i - 2].keys():
                        continue

                    # create history
                    history = (sentence[i], v, sentence[i - 1], u, sentence[i - 2], t, sentence[i + 1])
                    v_u_t_val = pred_hist(history, pre_trained_weights, feature2id)

                    if (u, v) not in u_v_val_dict.keys():
                        u_v_val_dict[u, v] = [(v_u_t_val, t)]
                    else:
                        u_v_val_dict[u, v].append((v_u_t_val, t))

                    if (t, u) not in u_t_val_dict.keys():
                        u_t_val_dict[t, u] = v_u_t_val
                    else:
                        u_t_val_dict[t, u] += v_u_t_val

        curr_pi_array = []

        for (u, v), val_t_array in u_v_val_dict.items():
            arr_val = [((val / u_t_val_dict[t, u]) * pi_dict_arr[i - 2][t, u][0], t) for val, t in val_t_array]
            max_p, max_t = max(arr_val, key=lambda tup: tup[0])
            curr_pi_array.append((u, v, max_p, max_t))

        # beam search
        if beam is not None:
            curr_pi_array.sort(key=lambda tup: tup[2], reverse=True)
            curr_pi_array = curr_pi_array[: beam]

        # convert to dict
        curr_pi_dict = {(u, v): (max_p, max_t) for (u, v, max_p, max_t) in curr_pi_array}
        pi_dict_arr.append(curr_pi_dict)

    last_pi_list = [(u, v, p) for (u, v), (p, _) in pi_dict_arr[-1].items()]
    t_1, t_2 = max(last_pi_list, key=lambda tup: tup[2])[0:2]
    pred.append(t_2), pred.append(t_1)
    for i in reversed(range(2, len(sentence) - 2)):
        cur_t = pi_dict_arr[i][t_1, t_2][1]
        pred.append(cur_t)
        t_1, t_2 = cur_t, t_1

    pred.reverse()
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


def find_history_prob(history, weights, feature2id):
    """
    :param history: {c_word, c_tag, p_word, p_tag, pp_word, pp_tag, n_word}
    :param weights: pre-trained weights
    :param feature2id: list of all the tags
    :return: the probability of the history
    """
    c_word, c_tag, p_word, p_tag, pp_word, pp_tag, n_word = history
    norm_sum = sum([pred_hist((c_word, t, p_word, p_tag, pp_word, pp_tag, n_word), weights, feature2id)
                    for t in feature2id.feature_statistics.tags])
    return pred_hist(history, weights, feature2id) / norm_sum


def find_sentence_prob(sentence, predictions, weights, feature2id):
    """
    :param sentence: list of word : [w-1=*, w0=*, w1, .., wn=~]
    :param predictions: list of tags : [t1, t2, .. tn]
    :param weights: optimal weight vector
    :param feature2id:
    :return: the probability of the prediction
    """
    assert len(sentence) - 3 == len(predictions), "length of sentence is longer then the predictions"

    mul_prob = 1

    for i in range(2, len(sentence) - 1):

        if i == 2:
            # default tag because the first two words are *
            default_tag = predictions[i - 2]
            history = (sentence[i], predictions[i - 2], sentence[i - 1], default_tag,
                       sentence[i - 2], default_tag, sentence[i + 1])
        elif i == 3:
            # default tag because the first two words are *
            default_tag = predictions[i - 2]
            history = (sentence[i], predictions[i - 2], sentence[i - 1], predictions[i - 3],
                       sentence[i - 2], default_tag, sentence[i + 1])
        else:
            history = (sentence[i], predictions[i - 2], sentence[i - 1], predictions[i - 3],
                       sentence[i - 2], predictions[i - 4], sentence[i + 1])

        mul_prob *= find_history_prob(history, weights, feature2id)

    return mul_prob


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
        print(f"The probability of the sentence is :{find_sentence_prob(sentence, pred, pre_trained_weights, feature2id)}")
        sentence = sentence[2:]
        for i in range(len(pred)):
            if i > 0:
                output_file.write(" ")
            output_file.write(f"{sentence[i]}_{pred[i]}")
        output_file.write("\n")
    output_file.close()
