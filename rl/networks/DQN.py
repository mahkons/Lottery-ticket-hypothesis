import torch
import torch.nn as nn

# beware, works only with batches
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class DQN(nn.Module):
    def __init__(self, state_sz, action_sz, layers_sz, image_input):
        super(DQN, self).__init__()
    
        if image_input:
            assert layers_sz is None, "layers configuration for image input is not enabled yet"
            layers = self.create_atari_layers(action_sz)
        else:
            layers = self.create_linear_layers(state_sz, action_sz, layers_sz)

        self.seq = nn.Sequential(*layers)

    def create_linear_layers(self, state_sz, action_sz, layers_sz):
        layers = list()
        in_sz = state_sz
        for sz in layers_sz:
            layers += [nn.Linear(in_sz, sz), nn.ReLU(inplace=True)]
            in_sz = sz
        layers.append(nn.Linear(in_sz, action_sz))
        return layers

    # TODO enable layers size configuration for image input
    def create_atari_layers(self, action_sz):
        layers = [
            nn.Conv2d(
                in_channels=4,
                out_channels=16,
                kernel_size=8,
                stride=4,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=4,
                stride=2,
            ),
            nn.ReLU(inplace=True),
            Flatten(),
            nn.Linear(2592, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, action_sz)
        ]
        return layers

    def forward(self, x):
        return self.seq(x)

