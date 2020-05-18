import torch
from multiprocessing import Pool
from functools import partial
import argparse
from copy import deepcopy
import random

from configs import Experiment, DQNConfig, ReplayMemoryConfig, AdamConfig
from envs import LunarLander, Breakout, Assault, Enduro, RoadRunner, SpaceInvaders, LunarLanderWithNoise, ImageShuffle
from params import LunarLanderConfig
from pruners import RewindWrapper, ERPruner, LayerwisePruner, GlobalPruner, FirstLayerPruner
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
        layers_sz = None,
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
        env = LunarLander,
        hyperparams = custom_params,
        stop_criterion = NoStop(),
        pruner = make_pruner(rewind_epoch=0, rescale=None, pruner_constructor=ERPruner, reinit_to_random=False),
    )

    for layers_sz in [[2048, 512]]:
        for repeat in range(4):
            random_seed = random.randint(0, 10**9)

            exp = deepcopy(custom_experiment)
            exp.logname = "EpicBigLL_ER_repeat_{}".format(repeat)
            exp.random_seed = random_seed

            exp.hyperparams.layers_sz = layers_sz

            exp_list.append(exp)


    torch.save(exp_list, "generated/exp_list")

if __name__ == "__main__":
    generate_experiments()
