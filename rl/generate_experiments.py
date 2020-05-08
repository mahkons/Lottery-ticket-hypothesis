import torch
from multiprocessing import Pool
from functools import partial
import argparse
from copy import deepcopy
import random

from configs import Experiment, DQNConfig, ReplayMemoryConfig, AdamConfig
from envs import LunarLander, Breakout, Assault, Enduro, RoadRunner, SpaceInvaders, LunarLanderWithNoise, ImageShuffle
from params import LunarLanderConfig
from pruners import RewindWrapper, ERPruner, LayerwisePruner, GlobalPruner
from pruners import L1GlobalRescale, L1LocalRescale, L2LocalRescale, L2GlobalRescale
from agent.stop_criterions import MaskDiffStop, EarlyBirdStop, NoStop, FixedEpochsStop
from launch_parallel import make_pruner


def generate_experiments():
    exp_list = list()
    custom_params = DQNConfig(
        memory_config = ReplayMemoryConfig(200*1000),
        optimizer_config = AdamConfig(1e-3),
        batch_size = 64,
        gamma = 0.99,
        eps_start = 0.9,
        eps_end = 0.05,
        eps_decay = 20000,
        target_net_update_steps = 2500,
        layers_sz = [],
        image_input = False,
        best_model_path = ":(",
    )

    custom_experiment = Experiment(
        opt_steps = 1000*1000,
        episodes = 10**10,
        prune_iters = 15,
        prune_percent = 20,
        device = None,
        logname = None,
        random_seed = None,
        env = Assault,
        hyperparams = custom_params,
        stop_criterion = FixedEpochsStop(limit_epochs=20),
        pruner = make_pruner(rewind_epoch=0, rescale=L2GlobalRescale(), pruner_constructor=GlobalPruner, reinit_to_random=False),
    )

    for env, name in [(partial(ImageShuffle, 2, (3, 1, 2, 0), Assault), "Shuffled2"),
            (partial(ImageShuffle, 4, (12, 2, 10, 11, 3, 9, 1, 4, 8, 0, 7, 6, 14, 15, 5, 13), Assault), "Shuffled4"),
            (Assault, "NotShuffled1"),
            (Assault, "NotShuffled2")]:
        random_seed = random.randint(0, 10**9)

        exp = deepcopy(custom_experiment)
        exp.logname = "Assault" + name + "Prune"
        exp.random_seed = random_seed

        exp.env = env
        
        exp_list.append(exp)


    torch.save(exp_list, "generated/exp_list")

if __name__ == "__main__":
    generate_experiments()
