import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.SparseLinear import SparseLinear


class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        #  self.fc1 = SparseLinear(16 * 5 * 5, 120)
        #  self.fc2 = SparseLinear(120, 84)
        #  self.fc3 = SparseLinear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def get_path(self):
        return SimpleNet.__name__;

    @classmethod
    def load_model(cls, path):
        state_dict = torch.load(path, map_location='cpu')
        net = cls()
        net.load_state_dict(state_dict=state_dict)
        return net

    def save_model(self, path):
        torch.save(self.state_dict(), path)
