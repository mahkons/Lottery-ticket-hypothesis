from pruners import GlobalPruner

class RescalingGlobalPruner(GlobalPruner):
    def reinit_net(self):
        super().reinit_net()
        for name, param in self.net.named_parameters():
            scaling_factor = param.data.sum() / self.net_init_state[name].sum()
            param.data *= scaling_factor
