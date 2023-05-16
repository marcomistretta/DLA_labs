from torch import nn
import torch.nn.functional as F


class DQLN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQLN, self).__init__()
        hidden_sizes = [128, 64]  # hidden layer sizes
        self.layer1 = nn.Linear(n_observations, hidden_sizes[0])
        self.layer2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.layer3 = nn.Linear(hidden_sizes[1], n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
