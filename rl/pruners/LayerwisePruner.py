import torch
import torch.nn as nn
import numpy as np

from pruners.Pruner import Pruner


# Prunes p% of remaining weights on each layer separately 
# prunes only weights, not biases
class LayerwisePruner(Pruner):
    def prune_net(self, p):
        for name, param in self.net.named_parameters():
            if not "weight" in name:
                continue

            data = param.data[self.mask[name] > 0.5]
            threshold = np.percentile(np.abs(data.cpu().numpy()), p)

            self.mask[name][torch.abs(param.data) < threshold] = 0
            param.data *= self.mask[name]
