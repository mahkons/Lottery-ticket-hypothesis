import torch
import torch.nn as nn
import numpy as np

from pruners.Pruner import Pruner


# Prunes p% of remaining weights
# prunes both weights and biases
class GlobalPruner(Pruner):
    def prune_net(self, p):
        full_data = list()
        for name, param in self.net.named_parameters():
            full_data.append(param.data[self.mask[name] > 0.5].reshape(-1))

        full_data = torch.abs(torch.cat(full_data))
        threshold = np.percentile(full_data.cpu().numpy(), p)

        for name, param in self.net.named_parameters():
            self.mask[name][torch.abs(param.data) < threshold] = 0
            param.data *= self.mask[name]

