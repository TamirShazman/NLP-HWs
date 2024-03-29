from scipy import sparse
from collections import OrderedDict, defaultdict
import numpy as np
from typing import List, Dict, Tuple
import added_features as f
from copy import deepcopy

WORD = 0
TAG = 1


class FeatureStatistics:
    def __init__(self, feature_dict_list=None):
        self.n_total_features = 0  # Total number of features accumulated

        # Init all features dictionaries
        if feature_dict_list == None:
            self.feature_dict_list = ["f100", "f101", "f102", "f103", "f104", "f105", "f106", "f107", "numbers", "capital_letters"]  # the feature classes used in the code
        else:
            self.feature_dict_list = feature_dict_list
        self.feature_rep_dict = {fd: OrderedDict() for fd in self.feature_dict_list}
        '''
        A dictionary containing the counts of each data regarding a feature class. For example in f100, would contain
        the number of times each (word, tag) pair appeared in the text.
        '''
        self.tags = set()  # a set of all the seen tags
        self.tags.add("~")
        self.tags_counts = defaultdict(int)  # a dictionary with the number of times each tag appeared in the text
        self.words_count = defaultdict(int)  # a dictionary with the number of times each word appeared in the text
        self.histories = []  # a list of all the histories seen at the test

    def get_word_tag_pair_count(self, file_path) -> None:
        """
            Extract out of text all word/tag pairs
            @param: file_path: full path of the file to read
            Updates the histories list
        """
        with open(file_path) as file:
            for line in file:
                if line[-1:] == "\n":
                    line = line[:-1]
                split_words = line.split(' ')
                previous_tag = "*"
                previous2_tag = "*"
                previous_word = "*"

                for word_idx in range(len(split_words)):
                    cur_word, cur_tag = split_words[word_idx].split('_')
                    lowercase_cur_word = cur_word.lower() # use lower case for f100-f107 to limit feature set size
                    self.tags.add(cur_tag)
                    self.tags_counts[cur_tag] += 1
                    self.words_count[cur_word] += 1

                    # f100 - add pairs of current word and current tag
                    f.f100_6_7(self.feature_rep_dict["f100"], word=lowercase_cur_word, current_tag=cur_tag)

                    # f101 - all suffixes and their current tag
                    f.f101(self.feature_rep_dict["f101"], lowercase_cur_word, cur_tag)
                    # f102 - all prefixes and their current tag
                    f.f102(self.feature_rep_dict["f102"], lowercase_cur_word, cur_tag)

                    # f103-f105 - add triplets, duos and singles of label sequences
                    f.f103_5(self.feature_rep_dict["f103"], self.feature_rep_dict["f104"], self.feature_rep_dict["f105"],
                             cur_tag, previous_tag, previous2_tag)

                    # f106 - add pairs of previous word and current tag
                    f.f100_6_7(self.feature_rep_dict["f106"], word=previous_word, current_tag=cur_tag)

                    # f107 - add pairs of next word and current tag
                    if word_idx == len(split_words) - 1: # skip last word because no next word
                        continue
                    else:
                        f.f100_6_7(self.feature_rep_dict["f107"], word=split_words[word_idx + 1].split('_')[0].lower(), current_tag=cur_tag)

                    # contains number
                    f.contains_number(self.feature_rep_dict["numbers"], cur_word, cur_tag)

                    # contains capital letter & not beginning of sentence
                    f.contains_uppercase(self.feature_rep_dict["capital_letters"], previous_word, cur_word, cur_tag)

                    previous2_tag = previous_tag
                    previous_tag = cur_tag
                    previous_word = lowercase_cur_word

                sentence = [("*", "*"), ("*", "*")]
                for pair in split_words:
                    sentence.append(tuple(pair.split("_")))
                sentence.append(("~", "~"))

                for i in range(2, len(sentence) - 1):
                    history = (
                        sentence[i][0], sentence[i][1], sentence[i - 1][0], sentence[i - 1][1], sentence[i - 2][0],
                        sentence[i - 2][1], sentence[i + 1][0])

                    self.histories.append(history)


class Feature2id:
    def __init__(self, feature_statistics: FeatureStatistics, threshold: int):
        """
        @param feature_statistics: the feature statistics object
        @param threshold: the minimal number of appearances a feature should have to be taken
        """
        self.feature_statistics = feature_statistics  # statistics class, for each feature gives empirical counts
        self.threshold = threshold  # feature count threshold - empirical count must be higher than this

        self.n_total_features = 0  # Total number of features accumulated

        #TODO: add other features
        # Init all features dictionaries
        # ["f100", "f101", "f102", "f103", "f104", "f105", "f106", "f107", "numbers", "capital_letters"]
        self.feature_to_idx = {fd: OrderedDict() for fd in feature_statistics.feature_rep_dict}
        # self.feature_to_idx = {
        #     "f100": OrderedDict(),
        # }
        self.represent_input_with_features = OrderedDict()
        self.histories_matrix = OrderedDict()
        self.histories_features = OrderedDict()
        self.small_matrix = sparse.csr_matrix
        self.big_matrix = sparse.csr_matrix

    def get_features_idx(self) -> None:
        """
        Assigns each feature that appeared enough time in the train files an idx.
        Saves those indices to self.feature_to_idx
        """
        for feat_class in self.feature_statistics.feature_rep_dict:
            if feat_class not in self.feature_to_idx:
                continue
            for feat, count in self.feature_statistics.feature_rep_dict[feat_class].items():
                if count >= self.threshold:
                    self.feature_to_idx[feat_class][feat] = self.n_total_features
                    self.n_total_features += 1
        print(f"you have {self.n_total_features} features!")

    def calc_represent_input_with_features(self) -> None:
        """
        initializes the matrices used in the optimization process - self.big_matrix and self.small_matrix
        """
        big_r = 0
        big_rows = []
        big_cols = []
        small_rows = []
        small_cols = []
        for small_r, hist in enumerate(self.feature_statistics.histories):
            for c in represent_input_with_features(hist, self.feature_to_idx):
                small_rows.append(small_r)
                small_cols.append(c)
            for r, y_tag in enumerate(self.feature_statistics.tags):
                demi_hist = (hist[0], y_tag, hist[2], hist[3], hist[4], hist[5], hist[6])
                self.histories_features[demi_hist] = []
                for c in represent_input_with_features(demi_hist, self.feature_to_idx):
                    big_rows.append(big_r)
                    big_cols.append(c)
                    self.histories_features[demi_hist].append(c)
                big_r += 1
        self.big_matrix = sparse.csr_matrix((np.ones(len(big_rows)), (np.array(big_rows), np.array(big_cols))),
                                            shape=(len(self.feature_statistics.tags) * len(
                                                self.feature_statistics.histories), self.n_total_features),
                                            dtype=bool)
        self.small_matrix = sparse.csr_matrix(
            (np.ones(len(small_rows)), (np.array(small_rows), np.array(small_cols))),
            shape=(len(
                self.feature_statistics.histories), self.n_total_features), dtype=bool)


def represent_input_with_features(history: Tuple, dict_of_dicts: Dict[str, Dict[Tuple[str, str], int]])\
        -> List[int]:
    """
        Extract feature vector in per a given history
        @param history: tuple{c_word, c_tag, p_word, p_tag, pp_word, pp_tag, n_word}
        @param dict_of_dicts: a dictionary of each feature and the index it was given
        @return a list with all features that are relevant to the given history
    """
    c_word, c_tag, p_word, p_tag, pp_word, pp_tag, n_word = history
    uppercase_c_word = deepcopy(c_word)
    c_word = c_word.lower()
    p_word = p_word.lower()
    n_word = n_word.lower()
    pp_word = pp_word.lower()

    # c_word = history[0]
    # c_tag = history[1]
    features = []
    # TODO: Add other features

    # # f100
    # if (c_word, c_tag) in dict_of_dicts["f100"]:
    #     features.append(dict_of_dicts["f100"][(c_word, c_tag)])

    # f100
    f.add_f100_6_7(features, dict_of_dicts["f100"], c_word, c_tag)

    # f101
    f.add_f101(features, dict_of_dicts["f101"], c_word, c_tag)

    # f102
    f.add_f102(features, dict_of_dicts["f102"], c_word, c_tag)

    # f103-5
    f.add_f103_5(features, dict_of_dicts["f103"], dict_of_dicts["f104"], dict_of_dicts["f105"], c_tag, p_tag, pp_tag)

    # f106
    f.add_f100_6_7(features, dict_of_dicts["f106"], p_word, c_tag)

    # f107
    f.add_f100_6_7(features, dict_of_dicts["f107"], n_word, c_tag)

    # contains number
    f.add_contains_number(features, dict_of_dicts["numbers"], c_word, c_tag)

    # contains uppercase & not beginning of sentence
    f.add_contains_uppercase(features, dict_of_dicts["capital_letters"], p_word, uppercase_c_word, c_tag)

    return features


def preprocess_train(train_path, threshold):
    # Statistics
    statistics = FeatureStatistics()
    statistics.get_word_tag_pair_count(train_path)

    # feature2id
    feature2id = Feature2id(statistics, threshold)
    feature2id.get_features_idx()
    feature2id.calc_represent_input_with_features()
    print(feature2id.n_total_features)

    for dict_key in feature2id.feature_to_idx:
        print(dict_key, len(feature2id.feature_to_idx[dict_key]))
    return statistics, feature2id


def read_test(file_path, tagged=True) -> List[Tuple[List[str], List[str]]]:
    """
    reads a test file
    @param file_path: the path to the file
    @param tagged: whether the file is tagged (validation set) or not (test set)
    @return: a list of all the sentences, each sentence represented as tuple of list of the words and a list of tags
    """
    list_of_sentences = []
    with open(file_path) as f:
        for line in f:
            if line[-1:] == "\n":
                line = line[:-1]
            sentence = (["*", "*"], ["*", "*"])
            split_words = line.split(' ')
            for word_idx in range(len(split_words)):
                if tagged:
                    cur_word, cur_tag = split_words[word_idx].split('_')
                else:
                    cur_word, cur_tag = split_words[word_idx], ""
                sentence[WORD].append(cur_word)
                sentence[TAG].append(cur_tag)
            sentence[WORD].append("~")
            sentence[TAG].append("~")
            list_of_sentences.append(sentence)
    return list_of_sentences
