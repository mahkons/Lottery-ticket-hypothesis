import torch
import torch.nn as nn
import numpy as np

from pruners.Pruner import Pruner


# Prunes p% of remaining weights on each layer separately 
# prunes only weights, not biases
class LayerwisePruner(Pruner):
    def get_mask_to_prune(self, p):
        assert p >= 0

        mask_to_prune = dict()
        for name, param in self.net.named_parameters():
            mask_to_prune[name] = self.mask[name].clone()
            if not "weight" in name:
                continue

            data = param.data[self.mask[name]]
            threshold = np.percentile(np.abs(data.cpu().numpy()), p)

            mask_to_prune[name][torch.abs(param.data) < threshold] = 0

        return mask_to_prune
