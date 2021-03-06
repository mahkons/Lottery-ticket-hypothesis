from configs import DQNConfig, ReplayMemoryConfig, AdamConfig

def CartPoleConfig():
    return DQNConfig(
        memory_config = ReplayMemoryConfig(20000),
        optimizer_config = AdamConfig(1e-3),
        batch_size = 64,
        gamma = 0.999,
        eps_start = 0.9,
        eps_end = 0.05,
        eps_decay = 1000,
        target_net_update_steps = 500,
        layers_sz = [64],
        image_input = False,

        best_model_path = ":(",
    )


def LunarLanderConfig():
    return DQNConfig(
        memory_config = ReplayMemoryConfig(200000),
        optimizer_config = AdamConfig(1e-3),
        batch_size = 64,
        gamma = 0.99,
        eps_start = 0.9,
        eps_end = 0.05,
        eps_decay = 20000,
        target_net_update_steps = 2500,
        layers_sz = [256, 128],
        image_input = False,

        #  best_model_path = "metrics/reference_models/LunarLander",
        best_model_path = ":(",
    )

def BigLunarLanderConfig():
    config = LunarLanderConfig()
    config.layers_sz = [2048, 512]
    config.best_model_path = ":("
    return config


def AtariConfig():
    return DQNConfig(
        memory_config = ReplayMemoryConfig(1000000),
        optimizer_config = AdamConfig(1e-5),
        batch_size = 64,
        gamma = 0.99,
        eps_start = 0.9,
        eps_end = 0.05,
        eps_decay = 100000,
        target_net_update_steps = 10000,
        layers_sz = "classic",
        image_input = True,

        best_model_path = ":(",
    )

def BigAtariConfig():
    config = AtariConfig()
    config.layers_sz = "big"
    return config
