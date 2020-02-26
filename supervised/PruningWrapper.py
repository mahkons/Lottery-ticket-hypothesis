import torch
import torch.nn as nn
import numpy as np

class PruningWrapper():
    def __init__(self, net, device):
        self.net = net.to(device)
        self.device = device

        self.net_init_state = dict()
        self.mask = dict()
        for name, param in net.named_parameters():
            self.net_init_state[name] = param.data.clone().detach()
            self.mask[name] = torch.ones(param.data.shape, device=device, requires_grad=False)

    def reinit_net(self):
        for name, param in self.net.named_parameters():
            param.data = self.net_init_state[name] * self.mask[name]

    def zero_unmasked_grad(self):
        for name, param in self.net.named_parameters():
            param.grad.data *= self.mask[name]

    def zero_unmasked(self):
        for name, param in self.net.named_parameters():
            param.data *= self.mask[name]

    def prune_net(self, p):
        for name, param in self.net.named_parameters():
            if not "weight" in name:
                continue

            data = param.data.cpu().numpy()[self.mask[name].cpu().numpy() > 0.5]
            threshold = np.percentile(np.abs(data), p)

            self.mask[name][torch.abs(param.data) < threshold] = 0
            param.data *= self.mask[name]


    def __call__(self, x):
        return self.net(x)

    def parameters(self):
        return self.net.parameters()

    def get_path(self):
        return self.net.get_path()

    def save_model(self, path):
        self.net.save_model(path)
        torch.save(self.net_init_state, path + "_net_init")
        torch.save(self.mask, path + "_mask")

    @classmethod
    def load_model(cls, netType, path, device):
        net_init_state = torch.load(path + "_net_init")
        mask = torch.load(path + "_mask")
        net = netType.load_model(path).to(device)

        wrapper = cls(net, device)
        wrapper.net_init_state = net_init_state
        wrapper.mask = mask

        return wrapper


class RescalingPruningWrapper(PruningWrapper):
    def reinit_net(self):
        super().reinit_net()
        for name, param in self.net.named_parameters():
            scaling_factor = self.net_init_state[name].abs().sum() / param.data.abs().sum()
            param.data *= scaling_factor

