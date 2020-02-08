from configs.Config import OptimizerConfig

class AdamConfig(OptimizerConfig):
    def __init__(self, lr: float):
        super(AdamConfig, self)
        self.lr = lr
