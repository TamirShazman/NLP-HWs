
import numpy as np

def nth_repl(string, substring, repl, n):
    """
    replace nth occurence of substring
    :param string: total string with substring that is replaced
    :param substring: target substring to be replaced
    :param repl: the word replacing the substring
    :param n: the occurence of the substring to be replaced
    :return:
    """
    find = string.find(substring)
    # If find is not -1 we have found at least one match for the substring
    i = find != -1
    # loop util we find the nth or we find no match
    while find != -1 and i != n:
        # find + 1 means we start searching from after the last match
        find = string.find(substring, find + 1)
        i += 1
    # If i is equal to n we found nth match so replace
    if i == n:
        return string[:find] + repl + string[find + len(substring):]
    return string

def dependency_postprocessing(sentences, real_roots, real_modifiers, spacy_processor):
    """
    Identify sentence spacy_root, replace if necessary with given roots.
    :param sentence:
    :param roots:
    :param modifiers:
    :param spacy_processor:
    :return:
    """

    spacy_doc = spacy_processor(sentences)
    spacy_roots = [sent.root for sent in spacy_doc.sents]
    spacy_childrens = [[child for child in sent.root.children] for sent in spacy_doc.sents]

    final_sentence = ''
    # for each sentence check if real root appears multiple times, replace only root with given real root.
    # Then replace children of spacy_root of the same POS with real modifers
    for sentence_count, (sentence, spacy_root, spacy_children, real_root, real_modifier) in enumerate(zip(spacy_doc.sents, spacy_roots, spacy_childrens, real_roots, real_modifiers)):
        # if the root is one of the modifiers then we want it to remain in the sentence
        if sentence.root.text.lower() in real_modifier or real_root.lower() + ' ' in sentence.text.lower() or ' ' + real_root.lower() in sentence.text.lower():
            replaced_sentence = sentence.text
        else:
            # the spacy_root part---------------------------------------------
            if sentence.text.count(spacy_root.text) > 1:
                i = 1
                for token in sentence:
                    if token.text == spacy_root.text and token.dep_ != "ROOT":
                        i = i + 1
                    elif token.dep_ == "ROOT":
                        break
                    else:
                        continue
                replaced_sentence = nth_repl(string=sentence.text, substring=spacy_root.text, repl=real_root, n=i)
            else:
                replaced_sentence = sentence.text.replace(spacy_root.text, real_root)
        final_sentence = final_sentence + ' ' + replaced_sentence

    # if the prediction has more sentences then should be
    final_sentence_count = len(final_sentence.split('.'))
    if len(sentences.split('.')) > final_sentence_count:
        # print(sentence_count, len(sentences.split('.')))
        for missed_sentence in sentences.split('.')[final_sentence_count-1:]:
            if len(missed_sentence) <= 1:
                continue
            else:
                final_sentence = final_sentence + missed_sentence + '.'
    return final_sentence[1:]


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels


def compute_metrics(eval_preds, tokenizer, metric):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_preds, labels = postprocess_text(decoded_preds, labels)

    # print('\n'+decoded_preds[0])
    result = metric.compute(predictions=decoded_preds, references=labels)

    result = {"bleu": result["score"]}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["avg_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result
