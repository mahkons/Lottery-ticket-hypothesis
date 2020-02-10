import torch
import torch.nn as nn
import numpy as np


class Pruner():
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

    def prune_net(self, p):
        raise NotImplementedError()
