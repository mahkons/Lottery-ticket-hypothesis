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
        memory_config = ReplayMemoryConfig(1000*1000),
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

    custom_experiment = Experiment(
        opt_steps = 500*1000,
        episodes = 10**10,
        prune_iters = 5,
        prune_percent = 30,
        device = None,
        logname = None,
        random_seed = None,
        env = LunarLanderWithNoise,
        hyperparams = custom_params,
        stop_criterion = FixedEpochsStop(limit_epochs=10),
        pruner = make_pruner(rewind_epoch=0, rescale=None, pruner_constructor=GlobalPruner, reinit_to_random=False),
    )

    for env, env_name in [(partial(ImageShuffle, 2, (3, 1, 2, 0), Assault), "AssaultShuffled2"),
            (partial(ImageShuffle, 4, [12,  2, 10, 11,  3,  9,  1,  4,  8,  0,  7,  6, 14, 15,  5, 13], Assault), "AssaultShuffled4")]:
        for layers_sz in ["classic", "big"]:
            random_seed = random.randint(0, 10**9)

            exp = deepcopy(custom_experiment)
            exp.logname = env_name + "_" + layers_sz
            exp.random_seed = random_seed

            exp.env = env
            exp.hyperparams.layers_sz = layers_sz
            
            exp_list.append(exp)


    torch.save(exp_list, "generated/exp_list")

if __name__ == "__main__":
    generate_experiments()
