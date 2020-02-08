import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, state_sz, action_sz, layers_sz):
        super(DQN, self).__init__()

        layers = list()
        
        in_sz = state_sz
        for sz in layers_sz:
            layers += [nn.Linear(in_sz, sz), nn.ReLU(inplace=True)]
            in_sz = sz
        layers.append(nn.Linear(in_sz, action_sz))

        self.seq = nn.Sequential(*layers)

    def forward(self, x):
        return self.seq(x)

