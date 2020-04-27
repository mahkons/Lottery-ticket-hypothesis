import torch.optim

from configs.Config import OptimizerConfig

class AdamConfig(OptimizerConfig):
    def __init__(self, lr: float):
        super(AdamConfig, self)
        self.lr = lr

    def create_optimizer(self, parameters):
        return torch.optim.Adam(parameters, lr=self.lr)
