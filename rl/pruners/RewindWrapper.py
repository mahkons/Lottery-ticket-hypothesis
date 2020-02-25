

# After reinit_epochs saves current weights
# Next pruner reinits will return to those weights
class RewindWrapper():
    def __init__(self, pruner, reinit_epochs):
        self.pruner = pruner
        self.reinit_epochs = reinit_epochs
        self.pruner_epoch = 0

    def epoch_step(self):
        self.pruner_epoch += 1
        if self.pruner_epoch == self.reinit_epochs:
            self.pruner.update_init_state()

    def reinit_net(self):
        # do not update pruner_epoch
        # rewind only once
        self.pruner.reinit_net()

    def optimization_step(self):
        self.pruner.reinit_net()

    def get_current_mask(self):
        return self.pruner.get_current_mask()

    def get_mask_to_prune(self, p):
        return self.pruner.get_mask_to_prune(p)

    def prune_net(self, p):
        self.pruner.prune_net(p)

