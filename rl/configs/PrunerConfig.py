from configs.Config import Config


class PrunerConfig(Config):
    def __init__(self, rewind_epochs):
        super(PrunerConfig, self).__init__()
        self.rewind_epochs = rewind_epochs
