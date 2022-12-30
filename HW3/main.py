from utils import *
from training import *

def main():
    training_path = 'data/train.labeled'
    val_path = 'data/test.labeled'
    comp_path = 'data/comp.unlabeled'

    # set training dataset
    training_s = get_tokens(training_path)
    training_pos = get_token_pos(training_path)
    training_true_tree = get_head_token(training_path)
    word2idx_sen, idx2word_sen = tokenize(training_s)
    word2idx_pos, idx2word_pos = tokenize(training_pos)
    training_x_s = convert_to_tokenized(training_s, word2idx_sen)
    training_x_pos = convert_to_tokenized(training_pos, word2idx_pos)
    training_len = [len(t) for t in training_true_tree]
    training_x_s_p = padding_(training_x_s, max(training_len))
    training_x_pos_p = padding_(training_x_pos, max(training_len))
    training_true_tree_p = padding_(training_true_tree, max(training_len))
    training_ds = ParserDataSet(training_x_s_p, training_x_pos_p, training_len, training_true_tree_p)

    # set validation dataset
    val_s = get_tokens(val_path)
    val_pos = get_token_pos(val_path)
    val_true_tree = get_head_token(val_path)
    val_x_s = convert_to_tokenized(val_s, word2idx_sen)
    val_x_pos = convert_to_tokenized(val_pos, word2idx_pos)
    val_len = [len(t) for t in val_true_tree]
    val_x_s_p = padding_(val_x_s, max(val_len))
    val_x_pos_p = padding_(val_x_pos, max(val_len))
    val_true_tree_p = padding_(val_true_tree, max(val_len))
    val_ds = ParserDataSet(val_x_s_p, val_x_pos_p, val_len, val_true_tree_p)

    # train
    training(training_ds, val_ds, idx2word_sen, idx2word_pos)

if __name__ == "__main__":
    main()