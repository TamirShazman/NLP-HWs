from training import *
import pickle

def main():
    training_path = 'data/train.labeled'
    val_path = 'data/test.labeled'
    tokenizer_path = 'embedding/tokenizer.pkl'
    idx2wordsen_path = 'embedding/idx2wordsen.pkl'
    tokenizer_pos_path = 'embedding/pos_tokenizer.pkl'
    idx2wordsen_pos_path = 'embedding/posidx2wordsen.pkl'
    comp_path = 'data/mytest.unlabeled'

    # set training dataset
    training_s = get_tokens(training_path) + get_tokens(val_path) + get_tokens('data/mytest2.unlabeled')
    training_pos = get_token_pos(training_path) + get_token_pos(val_path)+ get_token_pos('data/mytest2.unlabeled')
    training_true_tree = get_head_token(training_path) + get_head_token(val_path)+ get_head_token('data/mytest2.unlabeled')

    with open(tokenizer_path, 'rb') as f:
        word2idx_sen = pickle.load(f)
    with open(idx2wordsen_path, 'rb') as f:
        idx2word_sen = pickle.load(f)

    word2idx_pos, idx2word_pos = tokenize(get_token_pos(training_path) + get_token_pos(val_path))

    with open(tokenizer_pos_path, 'wb') as f:
        pickle.dump(word2idx_pos, f)
    with open(idx2wordsen_pos_path, 'wb') as f:
        pickle.dump(idx2word_pos, f)

    training_x_s = convert_to_tokenized(training_s, word2idx_sen)
    training_x_pos = convert_to_tokenized(training_pos, word2idx_pos)
    training_len = [len(t) for t in training_true_tree]
    training_x_s_p = padding_(training_x_s, max(training_len))
    training_x_pos_p = padding_(training_x_pos, max(training_len))
    training_true_tree_p = padding_(training_true_tree, max(training_len))
    training_ds = ParserDataSet(training_x_s_p, training_x_pos_p, training_len, training_true_tree_p)

    # set validation dataset
    val_s = get_tokens(comp_path)
    val_pos = get_token_pos(comp_path)
    val_true_tree = get_head_token(comp_path)
    val_x_s = convert_to_tokenized(val_s, word2idx_sen)
    val_x_pos = convert_to_tokenized(val_pos, word2idx_pos)
    val_len = [len(t) for t in val_true_tree]
    val_x_s_p = padding_(val_x_s, max(val_len))
    val_x_pos_p = padding_(val_x_pos, max(val_len))
    val_true_tree_p = padding_(val_true_tree, max(val_len))
    val_ds = ParserDataSet(val_x_s_p, val_x_pos_p, val_len, val_true_tree_p)

    # train
    training(training_ds, val_ds, idx2word_sen, idx2word_pos, w_embedding_dim=300, p_embedding_dim=30, hidden_dim=400, display_graph=True)
    #transformer_training(training_ds, val_ds, idx2word_sen, idx2word_pos, w_embedding_dim=300, p_embedding_dim=30)


if __name__ == "__main__":
    main()