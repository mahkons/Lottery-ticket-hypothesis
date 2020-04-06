import torch.nn as nn

# After reinit_epochs saves current weights
# Next pruner reinits will return to those weights
class RewindWrapper():
    def __init__(self, pruner, reinit_epochs, rescale=False, reinit_to_random=False):
        self.pruner = pruner
        self.reinit_epochs = reinit_epochs
        self.rescale = rescale
        self.reinit_to_random = reinit_to_random

        self.pruner_epoch = 0

    def epoch_step(self):
        self.pruner_epoch += 1
        if self.pruner_epoch == self.reinit_epochs:
            self.pruner.update_init_state()
        self.pruner.epoch_step()

    def reinit_net(self):
        # do not update pruner_epoch
        # rewind always to same epoch
        self.pruner.reinit_net()

        if self.reinit_to_random:
            for module in self.pruner.net.modules():
                if module == self.pruner.net or \
                    isinstance(module, nn.Sequential) or \
                    isinstance(module, nn.ReLU):
                    continue

                module.reset_parameters()

        if self.rescale:
            for name, param in self.pruner.net.named_parameters():
                scaling_factor = self.pruner.net_init_state[name].abs().sum() / param.data.abs().sum()
                param.data *= scaling_factor

    def optimization_step(self):
        self.pruner.optimization_step()

    def get_current_mask(self):
        return self.pruner.get_current_mask()

    def get_mask_to_prune(self, p):
        return self.pruner.get_mask_to_prune(p)

    def prune_net(self, p):
        self.pruner.prune_net(p)

    def get_all_weights(self):
        return self.pruner.get_all_weights()

