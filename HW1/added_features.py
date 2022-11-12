def f101(feature_dict, current_word, current_tag):
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


def f102(feature_dict, current_word, current_tag):
    if len(current_word) > 3:
        max_prefix = 4
    else:
        max_prefix = len(current_word)
    for i in range(max_prefix):
        prefix = current_word[:i]
        if (prefix, current_tag) not in feature_dict:
            feature_dict[(prefix, current_tag)] = 1
        else:
            feature_dict[(prefix, current_tag)] += 1

def f103_5(feature_dict_103, feature_dict_104, feature_dict_105, tag_i, tag_i_1, tag_i_2):
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

