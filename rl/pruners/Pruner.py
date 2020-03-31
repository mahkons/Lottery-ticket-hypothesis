import torch
import torch.nn as nn
import numpy as np

class Pruner():
    def __init__(self, net, device):
        self.net = net.to(device)
        self.device = device

        self.net_init_state = dict()
        self.mask = dict()
        self.update_init_state()

    def update_init_state(self):
        for name, param in self.net.named_parameters():
            self.net_init_state[name] = param.data.clone().detach()
            self.mask[name] = torch.ones(param.data.shape, dtype=torch.bool, device=self.device, requires_grad=False)

    def reinit_net(self):
        for name, param in self.net.named_parameters():
            param.data = self.net_init_state[name].clone().detach()
            param.data[~self.mask[name]] = 0

    def get_all_weights(self):
        full_data = list()
        for name, param in self.net.named_parameters():
            full_data.append(param.data[self.mask[name]].reshape(-1))
        return torch.cat(full_data)

    def optimization_step(self):
        self.zero_unmasked()

    def epoch_step(self):
        pass

    def zero_unmasked_grad(self):
        for name, param in self.net.named_parameters():
            param.grad.data[~self.mask[name]] = 0

    def zero_unmasked(self):
        for name, param in self.net.named_parameters():
            param.data[~self.mask[name]] = 0

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

