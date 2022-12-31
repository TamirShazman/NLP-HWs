from models import DependencyParser
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from chu_liu_edmonds import decode_mst
from utils import calculate_UAS

def training(training_ds, val_ds, idx2word_sen, idx2word_pos, w_embedding_dim = 30, p_embedding_dim = 30, hidden_dim = 50, batch_size=64, epochs=20):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    w_vocab_size = len(idx2word_sen)  # -2 because [UNK], [PAD]
    p_vocab_size = len(idx2word_pos)  # -2 because [UNK], [PAD]

    model = DependencyParser(w_vocab_size, p_vocab_size, w_embedding_dim, p_embedding_dim, hidden_dim)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    training_dl = DataLoader(training_ds, batch_size=batch_size)
    val_dl = DataLoader(val_ds, batch_size=batch_size)

    best_uas = 0
    best_epoch = None
    for epoch in range(epochs):
        print(f"\n -- Epoch {epoch} --")
        uas = train_epoch(model, device, optimizer, training_dl, val_dl)
        if uas > best_uas:
            best_uas = uas
            best_epoch = epoch



def train_epoch(model, device, optimizer, training_dl, val_dl):
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
                    loss, score_mat = model(sen, pos, true_tree)
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
                        loss, score_mat = model(sen, pos, true_tree)
                        batch_loss = batch_loss + loss
                        pred_tree, _ = decode_mst(score_mat.detach().cpu().numpy(), s_len.detach().cpu().numpy() , has_labels=False)
                        pred_trees.extend(pred_tree[1:])
                        gt_trees.extend(true_tree.detach().cpu().numpy().tolist()[1:])
                total_loss.append(batch_loss.detach().cpu().numpy().tolist())
                uas = calculate_UAS(pred_trees, gt_trees)

            t_bar.set_description(f"{phase}, loss: {sum(total_loss) / len(total_loss):.2f} UAS: {uas:.2f}")

    return uas
