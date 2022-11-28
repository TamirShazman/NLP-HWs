from torch import nn

class hw2_part2_model(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(input_size, int(input_size / 2)),
                                nn.ReLU(),
                                nn.Linear(int(input_size / 2), int(input_size / 4)),
                                nn.ReLU(),
                                nn.Linear(int(input_size / 4), int(input_size / 8)),
                                nn.ReLU())

    def forward(self, input):
        x = self.fc(input)
        return x