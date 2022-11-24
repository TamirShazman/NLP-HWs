import re
from gensim import downloader
import numpy as np
from torch.utils.data import Dataset
import os
from gensim.models import KeyedVectors

WORD_2_VEC_PATH = 'word2vec-google-news-300'
GLOVE_PATH = 'glove-twitter-200'


def download_model(model, path_to_word_rep):
    """
    :param path_to_word_rep: 
    :param model: glove or word2vec
    :return: download the weights of the model inside path_to_weights\ and return the path
    """
    path_to_weights = 'glove_weights.kv' if model == 'glove' else 'word2vec_weights.kv'

    if not os.path.isdir(path_to_word_rep):
        os.mkdir(path_to_word_rep)

    if os.path.isfile(os.path.join(path_to_word_rep, path_to_weights)):
        return os.path.join(path_to_word_rep, path_to_weights)

    model = downloader.load(GLOVE_PATH if model == 'glove' else WORD_2_VEC_PATH)
    model.save(os.path.join(path_to_word_rep, path_to_weights))

    return os.path.join(path_to_word_rep, path_to_weights)


def break_file_to_sentences(file_path):
    """
    :param file_path:
    :return: list of lists. for example the first list will contain all the words in the first sentence
    """
    list_of_sentences = []
    list_of_words = []

    with open(file_path) as f:
        for line in f:

            if line[-1:] == "\n":
                line = line[:-1]

            if line == '	' or len(line) == 0:
                list_of_sentences.append(list_of_words)
                list_of_words = []
                continue

            word_tagged = line.split('\t')

            if len(word_tagged) == 1:
                list_of_words.append(word_tagged)
            else:
                word, _ = word_tagged
                list_of_words.append(word)

    return list_of_sentences


def get_label(file_path):
    """
    :param file_path:
    :return: list of labels in order they appears
    """
    label_list = []
    with open(file_path) as f:
        for line in f:

            if line[-1:] == "\n":
                line = line[:-1]

            if line == '	' or len(line) == 0:
                continue

            word_tagged = line.split('\t')

            _, tag = word_tagged
            label_list.append(tag)
    return label_list


def convert_sentence_presentation_to_mean(sentence, model, window=1):
    """
    :param model: glove or word2vec
    :param window: number of words next to current word that will use in the mean
    :param sentence: list of words
    :return: np array list of size lxd, when l the sentence length and d is the embedded word presentation length.
    The word vector presentation in the returning array will be created with the mean of vectors inside the window.
    """
    assert window < len(sentence), "window size is too big"
    rep_matrix = []
    l = len(sentence)
    related_words = []
    for pos, word in enumerate(sentence):
        for i in range(pos - window, pos + window + 1):
            if 0 <= i <= l:
                related_words.append(sentence[i])
            else:
                continue

        vec = None
        for w in related_words:
            if w not in model.key_to_index:
                word = re.sub(r'\W+', '', word.lower())
                if word not in model.key_to_index:
                    continue
                if vec is None:
                    vec = model[w]
                else:
                    vec += model


class Dataset(Dataset):
    def __init__(self, file_path, model_weight_path):
        """
        :param file_path: path to the file
        :param model_weight_path: path to glove or word2vec saved weights
        """
        self.file_path = file_path
        self.words, self.tagges = self.extract_info()
        self.model = KeyedVectors.load(model_weight_path)
        self.word_rep_matrix, self.words_without_rep = self.find_word_rep()  # each row is a word in the corrsponding pos in words list

    def __len__(self):
        return len(self.words)

    def __getitem__(self, idx):
        if len(self.tagges) == 0:
            return self.get_item_no_tag(idx)
        else:
            return self.get_item_with_tag(idx)

    def get_item_no_tag(self, idx):
        word = self.words[idx]
        if word in self.words_without_rep:
            return None
        else:
            return self.word_rep_matrix[idx]

    def get_item_with_tag(self, idx):
        word, tag = self.words[idx], self.tagges[idx]
        if word in self.words_without_rep:
            return None, tag
        else:
            return self.word_rep_matrix[idx], tag

    def is_tagged(self):
        """
        :return: if the dataset is tagged
        """
        return len(self.tagges) != 0

    def extract_info(self):
        """
        :return: extract the words and the tagged (if there are)
        """
        list_of_words = []
        list_of_tagges = []

        with open(self.file_path) as f:
            for line in f:

                if line[-1:] == "\n":
                    line = line[:-1]

                if line == '	' or len(line) == 0:
                    continue

                word_tagged = line.split('\t')

                if len(word_tagged) == 1:
                    list_of_words.append(word_tagged)
                else:
                    word, tag = word_tagged
                    list_of_words.append(word)
                    list_of_tagges.append(0 if tag == 'O' else 1)

        assert len(list_of_tagges) != 0 and len(list_of_tagges) == len(list_of_words), 'Number of word and tags are ' \
                                                                                       'not equal '
        return list_of_words, list_of_tagges

    def find_word_rep(self):
        word_without_rep = []
        cur_rep = []
        for word in self.words:
            if word not in self.model.key_to_index:
                word = re.sub(r'\W+', '', word.lower())
                if word not in self.model.key_to_index:
                    word_without_rep.append(word)
                    print(f"Can't represent word {word}")
                else:
                    vec = self.model[word]
                    cur_rep.append(vec)
            else:
                vec = self.model[word]
                cur_rep.append(vec)

        return np.stack(cur_rep), word_without_rep
