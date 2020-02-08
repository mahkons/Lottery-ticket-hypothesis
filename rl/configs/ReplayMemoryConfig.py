from configs.Config import MemoryConfig

class ReplayMemoryConfig(MemoryConfig):
    def __init__(
            self,
            memory_size
        ):
        super(ReplayMemoryConfig, self).__init__()
        self.memory_size = memory_size
