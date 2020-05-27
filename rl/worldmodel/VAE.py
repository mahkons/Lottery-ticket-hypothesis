import typing

import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.transforms.functional import to_tensor

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def __init__(self, size, h, w):
        super().__init__()
        self.size = size
        self.h = h
        self.w = w

    def forward(self, input):
        return input.view(-1, self.size, self.h, self.w)


def calc_sz(h, k_sz, stride): 
    padding = []
    for k, s in zip(k_sz, stride):
        padding.append((h - k) % s)
        h = (h - k) // s + 1
    return h, padding


class VAE(nn.Module):
    def __init__(
            self,
            image_height: int,
            image_width: int,
            image_channels: int,
            z_dim: int,
            device: str = "cpu"
    ):
        super(VAE, self).__init__()
        self.device = device
        self.h = image_height
        self.w = image_width
        self.c = image_channels
        self.z_dim = z_dim

        k_sz = [4, 4, 4, 4]
        stride = [2, 2, 2, 2]

        hidden_h, padding_h = calc_sz(self.h, k_sz, stride)
        hidden_w, padding_w = calc_sz(self.w, k_sz, stride)
        h_dim = 256 * hidden_h * hidden_w
        output_padding = list(zip(padding_h, padding_w))

        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=k_sz[0], stride=stride[0]),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=k_sz[1], stride=stride[1]),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=k_sz[2], stride=stride[2]),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=k_sz[3], stride=stride[3]),
            nn.ReLU(),
            Flatten()
        )
        
        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)
        
        self.decoder = nn.Sequential(
            UnFlatten(256, hidden_h, hidden_w),
            nn.ConvTranspose2d(256, 128, kernel_size=k_sz[3], stride=stride[3], output_padding=output_padding[3]),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=k_sz[2], stride=stride[2], output_padding=output_padding[2]),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=k_sz[1], stride=stride[1], output_padding=output_padding[1]),
            nn.ReLU(),
            nn.ConvTranspose2d(32, image_channels, kernel_size=k_sz[0], stride=stride[0], output_padding=output_padding[0]),
            nn.Sigmoid(),
        )

        self.to(self.device)

    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def decode(self, z):
        z = self.fc3(z)
        z = self.decoder(z)
        return z

    def reparameterize(self, mu: torch.Tensor, logstd: torch.Tensor):
        if self.training:
            std = (logstd * 0.5).exp_()
            std_prob = torch.randn(*mu.size(), device=self.device)
            return mu + std_prob * std
        else:
            return mu   # inference time

    def forward(self, x: torch.Tensor):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        z = self.fc3(z)
        return self.decoder(z), mu, logvar

    @staticmethod
    def calculate_loss(pred_x: torch.Tensor, true_x: torch.Tensor, mu: torch.Tensor, logstd: torch.Tensor):
        bce = F.mse_loss(pred_x, true_x, reduction='sum')
        kld = -0.5 * torch.sum(1 + logstd - mu.pow(2) - logstd.exp())
        return bce + kld

    def play_encode(self, obs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.encode(obs)[0]

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    @classmethod
    def load_model(cls, path, *args, **kwargs):
        state_dict = torch.load(path, map_location='cpu')
        vae = cls(*args, **kwargs)
        vae.load_state_dict(state_dict=state_dict)
        return vae

