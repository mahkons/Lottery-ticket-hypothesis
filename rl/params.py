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

        best_model_path = "no:(",
    )


def LunarLanderConfig():
    return DQNConfig(
        memory_config = ReplayMemoryConfig(1000000),
        optimizer_config = AdamConfig(1e-3),
        batch_size = 64,
        gamma = 0.99,
        eps_start = 0.9,
        eps_end = 0.05,
        eps_decay = 5000,
        target_net_update_steps = 5000,
        layers_sz = [256, 128],
        image_input = False,

        best_model_path = "metrics/reference_models/LunarLander",
    )


def AtariConfig():
    return DQNConfig(
        memory_config = ReplayMemoryConfig(1000000),
        optimizer_config = AdamConfig(1e-3),
        batch_size = 64,
        gamma = 0.99,
        eps_start = 0.9,
        eps_end = 0.05,
        eps_decay = 5000,
        target_net_update_steps = 5000,
        layers_sz = [0], # tensorboardX fails to save with None or empty list hyperparam #TODO fix
        image_input = True,

        best_model_path = "no:(",
    )
