import torch
from multiprocessing import Pool
from functools import partial
import argparse
from copy import deepcopy
import random

from configs import Experiment, DQNConfig, ReplayMemoryConfig, AdamConfig
from envs import LunarLander
from params import LunarLanderConfig
from pruners import RewindWrapper, ERPruner, LayerwisePruner, GlobalPruner
from agent.stop_criterions import MaskDiffStop, EarlyBirdStop, NoStop
from launch_parallel import make_pruner


def generate_experiments():
    exp_list = list()
    custom_params = DQNConfig(
        memory_config = ReplayMemoryConfig(200000),
        optimizer_config = AdamConfig(1e-3),
        batch_size = 64,
        gamma = 0.99,
        eps_start = 0.9,
        eps_end = 0.05,
        eps_decay = 20000,
        target_net_update_steps = 5000,
        layers_sz = [256, 128],
        image_input = False,
        best_model_path = ":(",
    )

    custom_experiment = Experiment(
        opt_steps = 1000 * 1000,
        episodes = 10**10,
        prune_iters = 1,
        prune_percent = 0,
        device = None,
        logname = None,
        random_seed = None,
        env = LunarLander,
        hyperparams = custom_params,
        stop_criterion = MaskDiffStop(eps=0),
        pruner = make_pruner(rewind_epoch=0, rescale=False, pruner_constructor=LayerwisePruner),
    )

    for lr in [0.001]:
        for batch_size in [64]:
            for gamma in [0.99, 0.995, 0.999]:
                for eps_decay in [20000]:
                    for target_net_update_steps in [2500, 5000, 10000, 20000]:

                        for repeat in range(5):
                            random_seed = random.randint(0, 10**9)

                            exp = deepcopy(custom_experiment)
                            exp.logname = "search_lr={}_batch_size={}_gamma={}_epsdecay={}_targetnetupdate={}_repeat_{}". \
                                format(lr, batch_size, gamma, eps_decay, target_net_update_steps, repeat)
                            exp.random_seed = random_seed

                            exp.hyperparams.optimizer_config.lr = lr
                            exp.hyperparams.batch_size = batch_size
                            exp.hyperparams.gamma = gamma
                            exp.hyperparams.eps_decay = eps_decay
                            exp.hyperparams.target_net_update_steps = target_net_update_steps

                            exp_list.append(exp)


    torch.save(exp_list, "generated/exp_list")

if __name__ == "__main__":
    generate_experiments()
