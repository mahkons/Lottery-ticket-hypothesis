

# After reinit_epochs saves current weights
# Next pruner reinits will return to those weights
class RewindWrapper():
    def __init__(self, pruner, reinit_epochs, rescale=False):
        self.pruner = pruner
        self.reinit_epochs = reinit_epochs
        self.rescale = rescale

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

