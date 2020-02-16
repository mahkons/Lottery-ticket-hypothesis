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
            self.mask[name] = torch.ones(param.data.shape, dtype=torch.bool, device=device, requires_grad=False)

    def reinit_net(self):
        for name, param in self.net.named_parameters():
            param.data = self.net_init_state[name].clone().detach()
            param.data[~self.mask[name]] = 0

    def optimization_step(self):
        self.zero_unmasked_grad()

    def zero_unmasked_grad(self):
        for name, param in self.net.named_parameters():
            param.grad.data[~self.mask[name]] = 0

    def get_current_mask(self):
        return self.mask

    # returns dict layer_name->mask of weights, which
    # prune_net function going to prune on that iteration
    def get_mask_to_prune(self, p):
        raise NotImplementedError()

    # prunes net by p percent
    def prune_net(self, p):
        self.mask = self.get_mask_to_prune(p)
        for name, param in self.net.named_parameters():
            param.data[~self.mask[name]] = 0

