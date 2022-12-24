from torch import nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 64

class hw2_part2_model(nn.Module):
    def __init__(self, input_size, output_size=2):
        super(hw2_part2_model, self).__init__()
        self.fc = nn.Sequential(nn.Linear(input_size, int(input_size / 2)),
                                nn.ReLU(),
                                nn.Dropout(p=0.3),
                                nn.Linear(int(input_size / 2), int(input_size / 4)),
                                nn.ReLU(),
                                nn.Dropout(p=0.3),
                                nn.Linear(int(input_size / 4), int(input_size / 4)),
                                nn.ReLU(),
                                nn.Dropout(p=0.3),
                                nn.Linear(int(input_size / 4), int(input_size / 4)),
                                nn.ReLU(),
                                nn.Dropout(p=0.3),
                                nn.Linear(int(input_size / 4), int(input_size / 8)),
                                nn.ReLU(),
                                nn.Linear(int(input_size / 8), int(input_size / 8)),
                                nn.ReLU(),
                                nn.Dropout(p=0.3),
                                nn.Linear(int(input_size / 8), output_size),
                                )

    def forward(self, input):
        x = self.fc(input)
        return x


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()