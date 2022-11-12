from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay,accuracy_score, recall_score, precision_score, f1_score
import matplotlib.pyplot as plt

def get_ground_and_predicted(test_file, prediction_file):
    test_labels = list()
    predicted_labels = list()
    # extract test ground truth
    with open(test_file) as file:
        for line in file:
            if line[-1:] == "\n":
                line = line[:-1]
            split_words = line.split(' ')

            for word_idx in range(len(split_words)):
                _, cur_tag = split_words[word_idx].split('_')
                test_labels.append(cur_tag)

    # extract the predicted labels
    i=1
    with open(prediction_file) as file:
        for line in file:
            if line[-1:] == "\n":
                line = line[:-1]
            split_words = line.split(' ')
            try:
                for word_idx in range(len(split_words)):
                    _, cur_tag = split_words[word_idx].split('_')
                    predicted_labels.append(cur_tag)
            except:
                print(i, line)
            i = i + 1
    # TODO: REMOVE BEFORE SUBMITTING
    min_labels = min(len(test_labels), len(predicted_labels))
    test_labels = test_labels[:min_labels]
    predicted_labels = predicted_labels[:min_labels]
    return test_labels, predicted_labels,


def get_accuracy(ground_truth, predicted):
    return accuracy_score(ground_truth, predicted)

def show_confusion_matrix(ground_truth, predicted, grading_metric):
    labels = list(set(ground_truth))
    scores = grading_metric(ground_truth, predicted, average=None, labels=labels)
    top_ten_indices = sorted(range(len(scores)), key=lambda i: scores[i])[-10:]
    top_ten_labels = [labels[i] for i in top_ten_indices]
    cm = confusion_matrix(ground_truth, predicted, labels=top_ten_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels = top_ten_labels)
    disp.plot()
    plt.show()

def test():
    ground_truth, predicted = get_ground_and_predicted("C:\\Users\\dovid\\PycharmProjects\\NLP\\NLP-HWs\\HW1\\data\\test1.wtag",
                                                       'C:\\Users\\dovid\\PycharmProjects\\NLP\\NLP-HWs\\HW1\\predictions.wtag')
    print(get_accuracy(ground_truth, predicted))
    show_confusion_matrix(ground_truth, predicted, precision_score)

if __name__ == '__main__':
    test()





