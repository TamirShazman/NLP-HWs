from preprocessing import Dataset, download_model


def main():
    train_path = 'data/train.tagged'
    path_to_word_rep = 'word_rep/'

    model_path = download_model('word2vec', path_to_word_rep)
    my_dataset = Dataset(train_path, model_path)


if __name__ == '__main__':
    main()
