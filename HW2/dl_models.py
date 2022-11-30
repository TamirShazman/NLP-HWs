from torch import nn
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 64

class hw2_part2_model(nn.Module):
    def __init__(self, input_size, output_size=2):
        super(hw2_part2_model, self).__init__()
        self.fc = nn.Sequential(nn.Linear(input_size, int(input_size / 2)),
                                nn.ReLU(),
                                # nn.Dropout(p=0.3),
                                nn.Linear(int(input_size / 2), int(input_size / 4)),
                                nn.ReLU(),
                                nn.Linear(int(input_size / 4), int(input_size / 4)),
                                nn.ReLU(),
                                nn.Linear(int(input_size / 4), int(input_size / 4)),
                                nn.ReLU(),
                                # nn.Dropout(p=0.3),
                                nn.Linear(int(input_size / 4), output_size)
                                )

    def forward(self, input):
        x = self.fc(input)
        return x


class WeightedFocalLoss(nn.Module):
    "Non weighted version of Focal Loss"
    def __init__(self, alpha=.25, gamma=2):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = (torch.ones(batch_size, 2) * torch.tensor([alpha, 1-alpha])).to(device)
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        targets = targets.type(torch.long)
        # at = self.alpha.gather(0, targets.data.view(-1))
        at = self.alpha.gather(0, targets.data)
        pt = torch.exp(-BCE_loss)
        F_loss = at*(1-pt)**self.gamma * BCE_loss
        return F_loss.mean()