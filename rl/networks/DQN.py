import torch
import torch.nn as nn

from worldmodel.VAE import VAE

# beware, works only with batches
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class DQN(nn.Module):
    def __init__(self, state_sz, action_sz, layers_sz, image_input, device):
        super(DQN, self).__init__()
        self.layers_sz = layers_sz
        self.device = device
    
        if image_input:
            layers = self.create_atari_layers(action_sz, layers_sz)
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
    def create_atari_layers(self, action_sz, layers_sz):
        if layers_sz == "classic":
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
        elif layers_sz == "big":
            layers = [
                nn.Conv2d(
                    in_channels=4,
                    out_channels=16,
                    padding=24,
                    kernel_size=48,
                    stride=4,
                ),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    in_channels=16,
                    out_channels=32,
                    padding=16,
                    kernel_size=32,
                    stride=2,
                ),
                nn.ReLU(inplace=True),
                Flatten(),
                nn.Linear(4608, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, action_sz)
            ]
        elif layers_sz == "vae":
            vae = VAE.load_model("generated/vae.torch", image_height=84, image_width=84, image_channels=1, z_dim=256)
            self.encoder = vae.encoder
            self.fc1 = vae.fc1
            self.fc2 = vae.fc2
            layers = [
                nn.Linear(1024, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, action_sz),
            ]
            self.forward_cnt = 0
        else:
            raise ValueError("Incorrect layers_sz")
        return layers


    def reparameterize(self, mu: torch.Tensor, logstd: torch.Tensor):
        std = (logstd * 0.5).exp_()
        std_prob = torch.randn(*mu.size(), device=self.device)
        return mu + std_prob * std

    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        return z

    def forward(self, x):
        if self.layers_sz == "vae":
            x = self.encode(x.reshape(-1, 1, 84, 84))
            return self.seq(x.reshape(-1, 4 * 256))
        return self.seq(x)

