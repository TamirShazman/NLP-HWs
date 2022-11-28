from sklearn.neighbors import KNeighborsClassifier
import torch
from torch.utils.data import DataLoader
from torch import nn
from dl_models import hw2_part2_model
import time
from sklearn.metrics import f1_score
from IPython import display
import matplotlib.pyplot as plt
from my_data import my_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def knn_prediction(x_train, y_train, x_test, k=3):
    """
    :param x_train:
    :param y_train:
    :param x_test:
    :param k:
    :return:
    """
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(x_train, y_train)
    return neigh.predict(x_test)

def dl_prediction(x_train, y_train, x_dev, y_dev):
    """
    Load, train and predict using NN
    @param x_train:
    @param y_train:
    @param x_text:
    @return:
    """

    train_dl, dev_dl = dl_load(x_train, y_train, x_dev, y_dev)


def dl_load(x_train, y_train, x_dev, y_dev):
    """
    Pass data to dataloaders for ease of training
    @param x_train:
    @param y_train:
    @param x_dev:
    @param y_dev:
    @return:
    """
    train_dataloader = DataLoader(my_dataset(x_train, y_train), batch_size=64, shuffle=True)
    dev_dataloader = DataLoader(my_dataset(x_dev, y_dev), batch_size=64, shuffle=False)
    return train_dataloader, dev_dataloader

def dl_train(model, train_load, test_load, epochs, loss_f, optimizer, model_file_name):
    train_loss = list()
    train_accuracy = list()

    test_loss = list()
    test_accuracy = list()

    total_start_time = time.perf_counter()

    # save best model
    best_loss = 100

    # Early stopping
    last_loss = 100
    patience = 50
    triggertimes = 0
    stop_training = False

    # breakpoint()
    for epoch in range(epochs + 1):
        start_time = time.perf_counter()
        epoch_train_loss = list()
        train_total = 0
        train_correct = 0

        # train model
        model.train()
        for X, y in train_load:
            X = X.to(device)
            y = y.to(device)

            # forward pass
            output = model(X)
            loss = loss_f(output, y)

            loss.backward()
            # nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

            # get accuracy
            epoch_train_loss.append(loss.cpu().item())
            predicted = torch.argmax(output, 1)
            train_total += y.size(0)
            train_correct += (predicted == y).sum()

        train_loss.append(sum(epoch_train_loss) / len(epoch_train_loss))
        train_accuracy.append((train_correct / train_total).cpu())

        # test model
        display.clear_output()

        predictions = list()
        model.eval()
        with torch.no_grad():
            epoch_test_loss = list()
            test_total = 0
            test_correct = 0

            for X, y in test_load:
                X = X.to(device)
                y = y.to(device)

                output = model(X)
                epoch_test_loss.append(loss_f(output, y).cpu().item())

                predicted = torch.argmax(output, 1)

                predictions.extend(predicted.cpu().tolist())
                test_total += y.size(0)
                test_correct += (predicted == y).sum()

            current_loss = sum(epoch_test_loss) / len(epoch_test_loss)
            test_loss.append(current_loss)
            test_accuracy.append((test_correct / test_total).cpu())

        print("Epoch time is: {}".format(time.perf_counter() - start_time))
        print("Total time is: {}".format(time.perf_counter() - total_start_time))
        # display_progress(train_loss, test_loss, train_accuracy, test_accuracy, epoch)

        # # save the model every 5 epochs
        # if (epoch + 1) % 5 == 0 or epoch == epochs-1:
        # torch.save(model.state_dict(), model_file_name + '.pkl')
        #
        # if current_loss < best_loss:
        #     best_loss = current_loss
        #     torch.save({
        #         'epoch': epoch,
        #         'model_state_dict': model.state_dict(),
        #         'optimizer_state_dict': optimizer.state_dict(),
        #         'loss': current_loss}, model_file_name + '.pkl')
        # elif stop_training:
        #     torch.save({
        #         'epoch': epoch,
        #         'model_state_dict': model.state_dict(),
        #         'optimizer_state_dict': optimizer.state_dict(),
        #         'loss': current_loss}, model_file_name + '.pkl')
        #     break

        loss = (train_loss, test_loss)
        accuracy = (train_accuracy, test_accuracy)

    return predictions, loss, accuracy

def display_progress(train_loss, test_loss, train_accuracy, test_accuracy, epoch=None):
    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(16, 4))
    ax[0].plot(train_loss, label="train")
    ax[0].plot(test_loss, label="test")
    ax[0].legend()
    if epoch is not None:
        ax[0].set_title("Model Loss - Epoch {}".format(epoch))
    else:
        ax[0].set_title("Model Loss")

    ax[1].plot(train_accuracy, label="train")
    ax[1].plot(test_accuracy, label="test")
    # ax[1].axhline(y=0.87, color = "red", linestyle='--', label="87% accuracy")
    ax[1].legend()
    if epoch is not None:
        ax[1].set_title("Model Accuracy - Epoch {}".format(epoch))
    else:
        ax[1].set_title("Model Accuracy")
    plt.show()

def dl_prediction(x_train, y_train, x_dev, y_dev):
    # loader
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_loader, dev_loader = dl_load(x_train, y_train, x_dev, y_dev)

    # train model
    model = hw2_part2_model(input_size=200, output_size=2).to(device)
    epochs = 10
    lr = 0.01

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model_file_name = 'hw2_part2'
    print('number of parameters: ', sum(param.numel() for param in model.parameters()))

    predictions, loss, accuracy = \
        dl_train(model, train_loader, dev_loader, epochs, criterion, optimizer, model_file_name)

    print("finished running")
    plt.plot(range(len(loss)), loss)
    plt.savefig('loss_over_epochs.pdf')
    return predictions


