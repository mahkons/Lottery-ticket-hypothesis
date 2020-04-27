import torch
from multiprocessing import Pool
from functools import partial
import argparse
from copy import deepcopy
import random

from configs import Experiment, DQNConfig, ReplayMemoryConfig, AdamConfig
from envs import LunarLander, Breakout, Assault, Enduro, RoadRunner, SpaceInvaders, LunarLanderWithNoise
from params import LunarLanderConfig
from pruners import RewindWrapper, ERPruner, LayerwisePruner, GlobalPruner
from pruners import L1GlobalRescale, L1LocalRescale, L2LocalRescale, L2GlobalRescale
from agent.stop_criterions import MaskDiffStop, EarlyBirdStop, NoStop
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
        eps_decay = 50000,
        target_net_update_steps = 10000,
        layers_sz = [0],
        image_input = True,
        best_model_path = ":(",
    )

    custom_experiment = Experiment(
        opt_steps = 1000*1000,
        episodes = 10**10,
        prune_iters = 5,
        prune_percent = 30,
        device = None,
        logname = None,
        random_seed = None,
        env = Assault,
        hyperparams = custom_params,
        stop_criterion = MaskDiffStop(eps=0),
        pruner = make_pruner(rewind_epoch=0, rescale=None, pruner_constructor=GlobalPruner, reinit_to_random=False),
    )

    for pruner_c in [GlobalPruner, LayerwisePruner]:
        for rescale in [None, L2GlobalRescale()]:
            random_seed = random.randint(0, 10**9)

            exp = deepcopy(custom_experiment)
            exp.logname = "Assault_" + pruner_c.__name__ + "_" + ("L2GlobalRescale" if rescale is not None else "NoRescale")
            exp.random_seed = random_seed

            pruner = make_pruner(rewind_epoch=0, rescale=rescale, pruner_constructor=pruner_c, reinit_to_random=False)
            exp.pruner = pruner

            exp_list.append(exp)

    #  for repeat in range(4):
        #  random_seed = random.randint(0, 10**9)

        #  exp = deepcopy(custom_experiment)
        #  exp.logname = "BreakoutNewParams_repeat_{}". \
            #  format(repeat)
        #  exp.random_seed = random_seed

        #  exp_list.append(exp)


    torch.save(exp_list, "generated/exp_list")

if __name__ == "__main__":
    generate_experiments()
