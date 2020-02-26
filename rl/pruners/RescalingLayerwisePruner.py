from pruners import LayerwisePruner

class RescalingLayerwisePruner(LayerwisePruner):
    def reinit_net(self):
        super().reinit_net()
        for name, param in self.net.named_parameters():
            scaling_factor = self.net_init_state[name].abs().sum() / param.data.abs().sum()
            param.data *= scaling_factor
