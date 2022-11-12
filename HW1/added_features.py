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