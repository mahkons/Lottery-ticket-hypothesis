import torch.optim

from configs.Config import OptimizerConfig

class RMSPropConfig(OptimizerConfig):
    def __init__(self, lr, momentum, denominator, eps):
        super(RMSPropConfig, self)
        self.lr = lr
        self.momentum = momentum
        self.denominator = denominator # ?
        self.eps = eps

    def create_optimizer(self, parameters):
        return torch.optim.RMSprop(parameters, lr=self.lr, momentum=self.momentum, eps=self.eps)

