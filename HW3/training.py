from models import DependencyParser, TransformerModel, CBOW_Model
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from chu_liu_edmonds import decode_mst
from utils import calculate_UAS
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from utils import *
import torch.nn as nn

def display_progress(train_loss, test_loss, train_uas, test_uas, epoch=None):
    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(16, 4))
    ax[0].plot(train_loss, label="train")
    ax[0].plot(test_loss, label="test")
    ax[0].legend()
    if epoch is not None:
        ax[0].set_title("Model NLL Loss - Epoch {}".format(epoch))
    else:
        ax[0].set_title("Model NLL Loss")

    ax[1].plot(train_uas, label="train")
    ax[1].plot(test_uas, label="test")
    ax[1].legend()
    if epoch is not None:
        ax[1].set_title("Model UAS - Epoch {}".format(epoch))
    else:
        ax[1].set_title("Model UAS")
    plt.show()


def training(training_ds,
             val_ds,
             idx2word_sen,
             idx2word_pos,
             w_embedding_dim = 30,
             p_embedding_dim = 30,
             hidden_dim = 50,
             batch_size=64,
             epochs=50,
             early_stop=3,
             display_graph=False):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    w_vocab_size = len(idx2word_sen)  # -2 because [UNK], [PAD]
    p_vocab_size = len(idx2word_pos)  # -2 because [UNK], [PAD]

    model = DependencyParser(w_vocab_size, p_vocab_size, w_embedding_dim, p_embedding_dim, hidden_dim)
    #model = TransformerModel(w_vocab_size, p_vocab_size, 300, 20, 4, 100, 4, 0.2, task_type='Depndencyparsing').to(device)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    training_dl = DataLoader(training_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size)

    best_uas = 0
    best_epoch = None
    train_val_loss_epochs = {"train": [], "val": []}
    train_val_uas_epochs = {"train": [], "val": []}
    for epoch in range(epochs):
        print(f"\n -- Epoch {epoch} --")
        train_val_loss, train_val_uas = train_epoch(model, device, optimizer, training_dl, val_dl)
        if train_val_uas[1] > best_uas:
            torch.save(model.state_dict(), 'embedding/final_model.pt')
            best_uas = train_val_uas[1]
            best_epoch = epoch

        train_val_loss_epochs['train'].append(train_val_loss[0])
        train_val_loss_epochs['val'].append(train_val_loss[1])
        train_val_uas_epochs['train'].append(train_val_uas[0])
        train_val_uas_epochs['val'].append(train_val_uas[1])

        if epoch - best_epoch >= early_stop:
            print(f"Early stop, best UAS on validation: {best_uas:.2f}, in epoch: {best_epoch}")
            break

    if display_graph:
        display_progress(train_val_loss_epochs["train"],
                         train_val_loss_epochs["val"],
                         train_val_uas_epochs["train"],
                         train_val_uas_epochs["val"],
                         best_epoch
                         )





def train_epoch(model, device, optimizer, training_dl, val_dl):
    train_val_loss = []
    train_val_uas = []

    for phase in ["train", "val"]:
        pred_trees = []
        gt_trees = []
        total_loss = []

        if phase == "train":
            model.train(True)
        else:
            model.eval()

        dl  = training_dl if phase == "train" else val_dl
        t_bar = tqdm(dl)
        for sens, poss, s_lens, true_trees in t_bar:
            batch_loss = 0.
            # load all batch to device
            sens = sens.to(device)
            poss = poss.to(device)
            s_lens = s_lens.to(device)
            true_trees = true_trees.to(device)

            if phase == "train":
                # split batch
                for sen, pos, s_len, true_tree in zip(sens, poss, s_lens, true_trees):
                    # remove padding
                    sen = sen[:s_len]
                    pos = pos[:s_len]
                    true_tree = true_tree[:s_len]
                    loss, score_mat = model(sen, pos, true_tree_heads=true_tree)
                    #print(score_mat.shape, s_len.detach().cpu().numpy() - 1)
                    batch_loss = batch_loss + loss
                    pred_tree, _ = decode_mst(score_mat.detach().cpu().numpy(), s_len.detach().cpu().numpy() , has_labels=False)
                    pred_trees.extend(pred_tree[1:])
                    gt_trees.extend(true_tree.detach().cpu().numpy().tolist()[1:])


                total_loss.append(batch_loss.detach().cpu().numpy().tolist())
                batch_loss = batch_loss * (1 / s_lens.shape[0])
                uas = calculate_UAS(pred_trees, gt_trees)
                batch_loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            else:
                with torch.no_grad():
                    batch_loss = 0.
                    # split batch
                    for sen, pos, s_len, true_tree in zip(sens, poss, s_lens, true_trees):
                        # remove padding
                        sen = sen[:s_len]
                        pos = pos[:s_len]
                        true_tree = true_tree[:s_len]
                        # print(sen, pos, true_tree, s_len)
                        loss, score_mat = model(sen, pos, true_tree_heads=true_tree)
                        batch_loss = batch_loss + loss
                        pred_tree, _ = decode_mst(score_mat.detach().cpu().numpy(), s_len.detach().cpu().numpy() , has_labels=False)
                        pred_trees.extend(pred_tree[1:])
                        gt_trees.extend(true_tree.detach().cpu().numpy().tolist()[1:])
                total_loss.append(batch_loss.detach().cpu().numpy().tolist())
                uas = calculate_UAS(pred_trees, gt_trees)

            t_bar.set_description(f"{phase}, loss: {sum(total_loss) / len(total_loss):.2f} UAS: {uas:.2f}")

        train_val_loss.append(sum(total_loss) / len(total_loss))
        train_val_uas.append(calculate_UAS(pred_trees, gt_trees))

    return train_val_loss,train_val_uas


def train_embedding(word_dataset, idx2word_sen, early_stop=100, batch_size=128, epochs=100):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    w_vocab_size = len(idx2word_sen)

    model = CBOW_Model(w_vocab_size)

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss = nn.CrossEntropyLoss()
    training_dl = DataLoader(word_dataset, batch_size=batch_size, shuffle=True)

    best_acc = 0
    best_epoch = 0
    for epoch in range(epochs):
        print(f"\n -- Epoch {epoch} --")
        train_acc = train_epoch_word_rep(model, device, optimizer, training_dl, loss)
        if train_acc > best_acc:
            best_uas = train_acc
            best_epoch = epoch

        if epoch - best_epoch >= early_stop:
            print(f"Early stop, best UAS on validation: {best_acc:.2f}, in epoch: {best_epoch}")
            break

    emb = list(model.parameters())[0]
    torch.save(emb,  'embedding/mtembedding.pt')

def train_epoch_word_rep(model, device, optimizer, dl, floss):
    preds = []
    gt = []
    total_loss = []

    model.train(True)

    t_bar = tqdm(dl)
    for X_s, y_s, lens in t_bar:
        # load all batch to device
        X_s = X_s.to(device)
        y_s = y_s.to(device)

        # remove padding
        pred = model(X_s)
        loss = floss(pred, y_s)
        preds.extend(pred.detach().cpu().numpy())
        gt.extend(y_s.detach().cpu().numpy())


        total_loss.append(loss.detach().cpu().numpy().tolist())
        acc = 0
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        t_bar.set_description(f"loss: {sum(total_loss) / len(total_loss):.2f} ACC: {acc:.2f}")


    return acc

def cross_validation(**kwargs):
    uas_val = []
    kf = KFold(n_splits=5, random_state=42, shuffle=True)
    for i, (train_index, test_index) in enumerate(kf.split(kwargs['x_s_p'])):
        # get training data
        training_x_s_p = kwargs['x_s_p'][train_index]
        training_x_pos_p = kwargs['x_pos_p'][train_index]
        training_len = kwargs['len'][train_index]
        training_true_tree_p = kwargs['true_tree_p'][train_index]
        training_ds = ParserDataSet(training_x_s_p, training_x_pos_p, training_len, training_true_tree_p)

        # get validation data
        val_x_s_p = kwargs['x_s_p'][test_index]
        val_x_pos_p = kwargs['x_pos_p'][test_index]
        val_len = kwargs['len'][test_index]
        val_true_tree_p = kwargs['true_tree_p'][test_index]
        val_ds = ParserDataSet(val_x_s_p, val_x_pos_p, val_len, val_true_tree_p)


def train_transformer(model, dataloader, device, w_vocab_size):
    model.train()
    epochs = 500
    total_loss = 0
    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.AdamW(model.parameters(), lr=0.0001)

    for epoch in range(epochs):
        for sens, poss, s_lens, true_trees in dataloader:
            optim.zero_grad()
            input = sens.clone()
            src_mask = model.generate_square_subsequent_mask(sens.size(1))
            rand_value = torch.rand(sens.shape)
            rand_mask = (rand_value < 0.15) * (input != 1) * (input != 0)
            mask_idx = (rand_mask.flatten() == True).nonzero().view(-1)
            input = input.flatten()
            input2 = poss.flatten()
            input2[mask_idx] = 0
            input[mask_idx] = 103
            input = input.view(sens.size())
            input2 = input.view(poss.size())

            out = model(input.to(device), input2.to(device), src_mask.to(device))
            loss = criterion(out.view(-1, w_vocab_size), sens.view(-1).to(device))
            total_loss += loss
            loss.backward()
            optim.step()

        print("Epoch: {} -> loss: {}".format(epoch + 1, total_loss / (len(dataloader) * epoch + 1)))

def transformer_training(training_ds, val_ds, idx2word_sen, idx2word_pos, embedding_dim=200 ):
    nhid = 200  # the dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = 2  # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 2  # the number of heads in the multiheadattention models
    dropout = 0.2  # the dropout value
    training_dl = DataLoader(training_ds, batch_size=64, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=64)
    w_vocab_size = len(idx2word_sen)  # -2 because [UNK], [PAD]
    p_vocab_size = len(idx2word_pos)  # -2 because [UNK], [PAD]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TransformerModel(w_vocab_size, p_vocab_size, embedding_dim, nhead, nhid, nlayers, dropout).to(device)
    train_transformer(model, training_dl, device, w_vocab_size)

