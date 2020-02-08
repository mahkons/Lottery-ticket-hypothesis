from configs.Config import Config

class DQNConfig(Config):
    def __init__(
            self,
            memory_config,
            optimizer_config,
            batch_size,
            gamma,
            eps_start,
            eps_end,
            eps_decay,
            target_net_update_steps,
            layers_sz
        ):
        super(DQNConfig, self).__init__()
        self.memory_config = memory_config
        self.optimizer_config = optimizer_config
        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.target_net_update_steps = target_net_update_steps
        self.layers_sz = layers_sz
