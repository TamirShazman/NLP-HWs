from gensim.models import KeyedVectors
import torch
import copy

from preprocessing import download_model, break_file_to_sentences, find_word_rep
from LSTM_model import LSTMDatasetTest, LSTMNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

def stack_sen_label(model, sentences, labels):
    final_x = []
    final_y = []
    sentences_len = [len(s) for s in sentences]
    counter = 0
    sentence_mean_len = max(sentences_len) + 1
    for s in sentences:
        x = []
        y = []
        gap = sentence_mean_len - len(s)
        pad = ['pad'] * gap
        new_s = copy.copy(s)
        new_s.extend(pad)
        for w in new_s:
            word_rep = find_word_rep(w, model)
            if word_rep is None:
                x.append(model['nonsense'])
            else:
                x.append(word_rep)
            if w != 'pad':
                y.append(labels[counter])
                counter += 1
            else:
                y.append(0)
        final_x.append(x)
        final_y.append(y)
    return torch.tensor(final_x), torch.tensor(final_y), torch.tensor(sentences_len)

def add_prediction_to_test(predictions, file_path):
    counter = 0
    with open(file_path, 'r') as f1, open('test.tagged', 'w') as f2:
        for line in f1:
            if line == '	' or len(line) == 0 or line == '\n':
                f2.write(line)
                continue

            if line[-1:] == "\n":
                line = line[:-1]

            label = 'O' if predictions[counter] == 0 else '1'
            f2.write(line + '\t' + label + '\n')
            counter+=1

def main():
    test_path = 'data/test.untagged'
    model_path_lstm = 'weights/best_LSTM_1.pt'
    path_to_word_rep = 'word_rep/'

    model_path = download_model('word2vec', path_to_word_rep)
    model = KeyedVectors.load(model_path)

    # create training dataset
    sentences = break_file_to_sentences(test_path)
    X, _, sen_len = stack_sen_label(model, sentences, [0] * sum([len(s) for s in sentences]))
    train_dataset = LSTMDatasetTest(X, sen_len)
    model = LSTMNet(input_size=300)
    model.load_state_dict(torch.load(model_path_lstm))
    model.to(device)
    model.eval()
    predictions = []
    for x, sen_len in train_dataset:
        with torch.no_grad():
            X = x[:sen_len].to(device)
            output = model(X)
            predicted = torch.argmax(output, 1)
            predictions.extend(predicted.cpu().tolist())
    add_prediction_to_test(predictions, test_path)


if __name__ == '__main__':
    main()
