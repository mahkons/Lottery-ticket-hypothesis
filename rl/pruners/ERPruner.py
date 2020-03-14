import torch
import torch.nn as nn
import numpy as np

from pruners.Pruner import Pruner


# Prunes p% of remaining weights
# prunes only weights
# sparsity in layer depends on layer size
# supposed to use with linear layers only!!!
class ERPruner(Pruner):
    def __init__(self, net, device):
        super(ERPruner, self).__init__(net, device)

        for module in net.modules():
            assert module == net or \
                isinstance(module, nn.Linear) or \
                isinstance(module, nn.ReLU) or \
                isinstance(module, nn.Sequential)

        self.density = 1

        params_sum = 0
        params_wsum = 0
        self.param_c = dict()
        for name, param in self.net.named_parameters():
            if not "weight" in name:
                continue
            n, m = param.data.shape
            params_sum += n * m
            self.param_c[name] = 1 - (n + m) / (n * m)
            params_wsum += n * m * self.param_c[name]

        self.s0 = params_sum / params_wsum


    def get_mask_to_prune(self, p):
        new_density = self.density * (1 - p/100.0)

        mask_to_prune = dict()
        for name, param in self.net.named_parameters():
            mask_to_prune[name] = self.mask[name].clone()
            if not "weight" in name:
                continue

            density = 1 - self.param_c[name] * (1 - new_density) * self.s0

            data = param.data
            threshold = np.percentile(np.abs(data.cpu().numpy()), density * 100)

            mask_to_prune[name][torch.abs(param.data) < threshold] = 0

        return mask_to_prune


    def prune_net(self, p):
        super().prune_net(p)
        self.density *= (1 - p/100.0)
    
