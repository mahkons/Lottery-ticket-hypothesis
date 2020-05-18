import torch
import torch.nn as nn
import numpy as np

from pruners.Pruner import Pruner


# Prunes p% of remaining weights from first layer
# prunes both weights and biases
class FirstLayerPruner(Pruner):
    def get_mask_to_prune(self, p):
        assert p >= 0

        net_params = list(self.net.named_parameters())

        full_data = list()
        for name, param in net_params[:2]:
            full_data.append(param.data[self.mask[name]].reshape(-1))

        full_data = torch.abs(torch.cat(full_data))
        threshold = np.percentile(full_data.cpu().numpy(), p)

        mask_to_prune = dict()

        for name, param in net_params:
            mask_to_prune[name] = self.mask[name].clone()

        for name, param in net_params[:2]:
            mask_to_prune[name][torch.abs(param.data) < threshold] = 0

        return mask_to_prune
    
