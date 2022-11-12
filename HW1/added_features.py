def f100_6_7(feature_dict, word, current_tag):
    # add pairs of word and current tag
    if (word, current_tag) not in feature_dict:
        feature_dict[(word, current_tag)] = 1
    else:
        feature_dict[(word, current_tag)] += 1

def add_f100_6_7(feature_list, feature_dict, word, current_tag):
    # add pairs of word and current tag
    if (word, current_tag) in feature_dict:
        feature_list.append(feature_dict[(word, current_tag)])

def f101(feature_dict, current_word, current_tag):
    # add suffixes up to a length of 4
    if len(current_word) > 3:
        max_suffix = 4
    else:
        max_suffix = len(current_word)
    for i in range(1, max_suffix + 1):
        suffix = current_word[-i:]
        if (suffix, current_tag) not in feature_dict:
            feature_dict[(suffix, current_tag)] = 1
        else:
            feature_dict[(suffix, current_tag)] += 1

def add_f101(feature_list, feature_dict, current_word, current_tag):
    # add suffixes up to a length of 4
    if len(current_word) > 3:
        max_suffix = 4
    else:
        max_suffix = len(current_word)
    for i in range(1, max_suffix + 1):
        suffix = current_word[-i:]
        if (suffix, current_tag) in feature_dict:
            feature_list.append(feature_dict[(suffix, current_tag)])

def f102(feature_dict, current_word, current_tag):
    # add prefixes up to a length of 4
    if len(current_word) > 3:
        max_prefix = 4
    else:
        max_prefix = len(current_word)
    for i in range(max_prefix):
        prefix = current_word[:i+1]
        if (prefix, current_tag) not in feature_dict:
            feature_dict[(prefix, current_tag)] = 1
        else:
            feature_dict[(prefix, current_tag)] += 1

def add_f102(feature_list, feature_dict, current_word, current_tag):
    # add prefixes up to a length of 4
    if len(current_word) > 3:
        max_prefix = 4
    else:
        max_prefix = len(current_word)
    for i in range(max_prefix):
        prefix = current_word[:i + 1]
        if (prefix, current_tag) in feature_dict:
            feature_list.append(feature_dict[(prefix, current_tag)])


def f103_5(feature_dict_103, feature_dict_104, feature_dict_105, tag_i, tag_i_1, tag_i_2):
    # CONTEXT FEATURES
    # add triplets of current, previous and penultimate tags
    if (tag_i, tag_i_1, tag_i_2) not in feature_dict_103:
        feature_dict_103[(tag_i, tag_i_1, tag_i_2)] = 1
    else:
        feature_dict_103[(tag_i, tag_i_1, tag_i_2)] += 1

    # add duos of current, previous tags
    if (tag_i, tag_i_1) not in feature_dict_104:
        feature_dict_104[(tag_i, tag_i_1)] = 1
    else:
        feature_dict_104[(tag_i, tag_i_1)] += 1

    # count how many tags appeared
    if tag_i not in feature_dict_105:
        feature_dict_105[tag_i] = 1
    else:
        feature_dict_105[tag_i] += 1

def add_f103_5(feature_list, feature_dict_103, feature_dict_104, feature_dict_105, tag_i, tag_i_1, tag_i_2):
    # add triplets of current, previous and penultimate tags
    if (tag_i, tag_i_1, tag_i_2) in feature_dict_103:
        feature_list.append(feature_dict_103[(tag_i, tag_i_1, tag_i_2)])

    # add duos of current, previous tags
    if (tag_i, tag_i_1) in feature_dict_104:
        feature_list.append(feature_dict_104[(tag_i, tag_i_1)])

    # count how many tags appeared
    if tag_i in feature_dict_105:
        feature_list.append(feature_dict_105[(tag_i)])

def contains_number(feature_dict, current_word, current_tag):
    # adds count if current tag is for a word with a number in it
    if any(char.isdigit() for char in current_word):
        if current_tag not in feature_dict:
            feature_dict[current_tag] = 1
        else:
            feature_dict[current_tag] += 1

def add_contains_number(feature_list, feature_dict, current_word, current_tag):
    # adds count if current tag is for a word with a number in it
    if any(char.isdigit() for char in current_word):
        if current_tag in feature_dict:
            feature_list.append(feature_dict[current_tag])

def contains_uppercase(feature_dict, previous_word, current_word, current_tag):
    # adds count if current tag is for a word with an upper case letter in it
    if previous_word == "*":
        return

    if any(char.isupper() for char in current_word):
        if current_tag not in feature_dict:
            feature_dict[current_tag] = 1
        else:
            feature_dict[current_tag] += 1

def add_contains_uppercase(feature_list, feature_dict, previous_word, current_word, current_tag):
    # adds count if current tag is for a word with a capitalized letter in it that is not the beginning of a sentence
    if previous_word == "*":
        return

    if any(char.isupper() for char in current_word):
        if current_tag in feature_dict:
            feature_list.append(feature_dict[current_tag])
