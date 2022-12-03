import re
from gensim import downloader
import numpy as np
from torch.utils.data import Dataset
import os
from gensim.models import KeyedVectors

WORD_2_VEC_PATH = 'word2vec-google-news-300'
GLOVE_TWITTER_PATH = 'glove-twitter-200'
GLOVE_WIKI_PATH = 'glove-wiki-gigaword-100'


def download_model(model, path_to_word_rep):
    """
    :param path_to_word_rep:
    :param model: glove_twiter or glove_wiki or word2vec
    :return: download the weights of the model inside path_to_weights\ and return the path
    """

    path_to_weights = 'glove_weights.kv' if model == 'glove' else 'word2vec_weights.kv'

    if not os.path.isdir(path_to_word_rep):
        os.mkdir(path_to_word_rep)

    if os.path.isfile(os.path.join(path_to_word_rep, path_to_weights)):
        return os.path.join(path_to_word_rep, path_to_weights)

    model = downloader.load(GLOVE_TWITTER_PATH if model == 'glove' else WORD_2_VEC_PATH)
    model.save(os.path.join(path_to_word_rep, path_to_weights))

    return os.path.join(path_to_word_rep, path_to_weights)


def break_file_to_sentences(file_path):
    """
    :param file_path:
    :return: list of lists. for example the first list will contain all the words in the first sentence
    """
    list_of_sentences = []
    list_of_words = []
    counter_1 = []
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
                word_tagged = line.split(' ')

            if len(word_tagged) == 1:
                list_of_words.append(word_tagged[0])
            else:
                word, _ = word_tagged
                list_of_words.append(word)
                counter_1.append(word)

    if len(list_of_words) > 0:
        list_of_sentences.append(list_of_words)

    return list_of_sentences


def get_label(file_path):
    """
    :param file_path:
    :return: list of labels in order they appears
    """
    label_lists = []
    with open(file_path) as f:
        for line in f:
            if line[-1:] == "\n":
                line = line[:-1]

            if line == '	' or len(line) == 0:
                continue

            word_tagged = line.split('\t')
            if len(word_tagged) == 1:
                word_tagged = line.split(' ')
            try:
                _, tag = word_tagged
            except:
                test = word_tagged
                print(test)

            label_lists.append(0 if tag == 'O' else 1)
    return label_lists


def find_word_rep(word, model):
    """
    :param word:
    :param model:
    :return: trying to find the word vector representation. this function might cut the word on do some changes in the
    word representation.
    :Note: You can add more rules if you find it useful
    """
    word = word.lower()

    if word in model.key_to_index:
        return model[word]

    word_tmp = re.sub(r'\W+', '', word.lower())
    if word_tmp in model.key_to_index:
        return model[word_tmp]

    if re.match(r'^[A-Za-z]+$', word):
        word_tmp = word.lower()
        for i in range(len(word_tmp)):
            beg_word = word_tmp[:i]
            end_word = word_tmp[i:]
            if beg_word in model.key_to_index and end_word in model.key_to_index:
                return model[beg_word] + model[end_word]

    if word.isnumeric():
        return model['number']

    if "http" in word:
        return model['http']

    if "@" in word:
        return model["@"]

    # check if word contain only one symol
    if word == len(word) * word[0] and word[0] in model.key_to_index:
        return model[word[0]]

    # find cases as 6th
    if re.match(r'^\d+th$', word):
        return model['number'] + model['th']

    # find cases as 23rd
    if re.match(r'^\d+rd$', word):
        return model['number'] + model['rd']

    # find cases as 2nd
    if re.match(r'^\d+nd$', word):
        return model['number'] + model['nd']

    # find for 6pm
    if re.match(r'^\d+pm$', word):
        return model['number'] + model['evening']

    # find for 6am
    if re.match(r'^\d+am$', word):
        return model['number'] + model['morning']

    # smily
    if word == ':)':
        return model['smily']

    # blink smily
    if word == ';-)' or word == ';)':
        return model['blink'] + model['smily']

    # sad smily
    if word == ':(' or word == ":'" or word == '=/' or word == ':/':
        return model['sad'] + model['smily']

    # decimal number
    if re.match(r'^\d*\.\d+$', word):
        return model['decimal']

    # too many letters
    if re.match(r'.*(\w)\1{2,}.*', word):
        dup_letter = re.findall(r'(\w)\1{2,}', word)[0]
        new_word = re.sub(r'(\w)\1{2,}', dup_letter, word)
        if new_word in model.key_to_index:
            return model[new_word]

    if '#' in word:
        return model['#']

    if '?' in word:
        # for differnt models
        if '?' in model.key_to_index:
            return model['?']
        else:
            return model['question'] + model['mark']

    # try to break word to two words.
    words_founded = []
    current_word = None
    start = 0
    i = 1
    while i <= len(word):
        tmp_word = word[start:i]

        if tmp_word in model.key_to_index:
            current_word = tmp_word
            i += 1

        elif current_word is not None:
            words_founded.append(current_word)
            start = i - 1
            current_word = None

        else:
            i += 1

    # check for last word
    if current_word in model.key_to_index:
        words_founded.append(current_word)

    # return if all the word founded
    if len(words_founded) != 0 and ''.join(words_founded) == word:
        rep_vec = 0
        for w in words_founded:
            rep_vec += model[w]
        return rep_vec

    # specific case
    if word == '3d':
        return model['three'] + model['dimensional']

    if 'ps' in word:
        return model['playstation']

    if 'iphone' in word:
        return model['cellphone']

    return None


def convert_sentence_presentation_to_mean(sentence, model, window=3, weight_word='weighted', use_pos_embeded=False):
    """
    :param use_pos_embeded: if position will be represented in the vector
    :param weight_word: in which why to weight the mean
    :param model: glove or word2vec
    :param window: number of words next to current word that will use in the mean
    :param sentence: list of words
    :return: np array list of size lxd, when l the sentence length and d is the embedded word presentation length.
    The word vector presentation in the returning array will be created with the mean of vectors inside the window.
    """
    count = 0
    rep_matrix = []
    # append start and end tokens
    sentence.insert(0, '*')
    sentence.insert(len(sentence), '~')

    l = len(sentence)

    for pos, word in enumerate(sentence):
        # start and end token
        if pos == 0 or pos == l - 1:
            continue
        related_words = []
        for i in range(pos - window, pos + window + 1):
            if 0 <= i <= l - 1:
                related_words.append(sentence[i])
            else:
                continue

        #  if note been given will compute normal average
        weight_word_dict = {}
        if weight_word == 'normal':
            weight_word_dict = {w: 1 for w in related_words}
        elif weight_word == 'weighted':
            # hyperparameter
            curr_word_weight = 0.95
            other_word_weight = (1 - curr_word_weight) / (len(related_words) - 1)
            for w in related_words:
                if w == word:
                    weight_word_dict[w] = curr_word_weight
                else:
                    weight_word_dict[w] = other_word_weight

        vec = 0
        num_of_founded_rep = 0
        for w in related_words:

            rep = find_word_rep(w, model)

            # if not a single representation found place a rare word
            if rep is None:
                count += 1
                rep = model['nonsense']

            num_of_founded_rep += 1
            vec += weight_word_dict[w] * rep
        vec = vec / num_of_founded_rep

        if use_pos_embeded:
            vec = np.concatenate((vec, find_word_rep(str(pos), model)))
        rep_matrix.append(vec)

    return np.stack(rep_matrix)


def convert_sentence_presentation_to_concatenation(sentence, model, window=1, use_pos_embeded=False):
    """
    :param use_pos_embeded: if position will be represented in the vector
    :param weight_word: in which why to weight the mean
    :param model: glove or word2vec
    :param window: number of words next to current word that will use in the mean
    :param sentence: list of words
    :return: np array list of size lxd, when l the sentence length and d is the embedded word presentation length.
    The word vector presentation in the returning array will be created with the mean of vectors inside the window.
    """

    rep_matrix = []
    # append start and end tokens
    for i in range(window):
        sentence.insert(0, '*')
    for i in range(window):
        sentence.insert(len(sentence), '~')

    l = len(sentence)

    for pos, word in enumerate(sentence):
        # start and end token
        if pos < window or pos > l - window - 1:  # pos == l - 1:
            continue
        related_words = []
        for i in range(pos - window, pos + window + 1):
            if 0 <= i <= l - 1:
                related_words.append(sentence[i])
            else:
                continue

        vec_parts = list()
        for w in related_words:

            rep = find_word_rep(w, model)

            # if not a single representation found place a rare word
            if rep is None:
                rep = model['nonsense']

            vec_parts.append(rep)
        vec = np.concatenate(vec_parts, axis=0)

        if use_pos_embeded:
            vec = np.concatenate((vec, find_word_rep(str(pos), model)))
        rep_matrix.append(vec)

    return np.stack(rep_matrix)


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
