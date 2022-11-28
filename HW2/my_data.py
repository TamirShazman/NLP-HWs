from torch.utils.data import Dataset
import torch

class my_dataset(Dataset):
    def __init__(self, X, y):
        self.features = X
        self.labels = y

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        x = torch.from_numpy(self.features[idx, :])
        y = self.labels[idx]
        return x, y
