import torch.nn.functional as F
from torch import nn


class DQN(nn.Module):
    def __init__(self, num_features, num_actions):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(num_features, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, num_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)