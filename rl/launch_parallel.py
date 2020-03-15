import torch
from multiprocessing import Pool
from functools import partial

from train import start_experiment
from configs import Experiment
from envs import LunarLander
from params import LunarLanderConfig
from pruners import RewindWrapper, ERPruner, LayerwisePruner, GlobalPruner
from agent.stop_criterions import MaskDiffStop, EarlyBirdStop, NoStop

device = torch.device('cpu')
exp_list = list()

def make_pruner__(rewind_epoch, rescale, pruner_constructor, net):
    return RewindWrapper(pruner_constructor(net, device), rewind_epoch, rescale=rescale)

def make_pruner(rewind_epoch, rescale, pruner_constructor):
    return partial(make_pruner__, rewind_epoch, pruner_constructor, rescale=rescale)


RANDOM_SEED=2020
exp_list.append(Experiment(
        opt_steps = 1000,
        episodes = 10**10,
        prune_iters = 2,
        prune_percent = 20,
        device = device,
        logname = "E1_test_v3",
        random_seed = RANDOM_SEED,
        env = LunarLander(random_state=RANDOM_SEED),
        hyperparams = LunarLanderConfig(),
        stop_criterion = MaskDiffStop(eps=0),
        pruner = make_pruner(0, ERPruner, False),
    )
)

RANDOM_SEED=2021
exp_list.append(Experiment(
        opt_steps = 1000,
        episodes = 10**10,
        prune_iters = 2,
        prune_percent = 20,
        device = device,
        logname = "E2_test_v3",
        random_seed = RANDOM_SEED,
        env = LunarLander(random_state=RANDOM_SEED),
        hyperparams = LunarLanderConfig(),
        stop_criterion = MaskDiffStop(eps=0),
        pruner = make_pruner(0, ERPruner, False),
    )
)


def launch_experiments():
    with Pool(processes=4) as pool:
        pool.map(start_experiment, exp_list)

if __name__ == "__main__":
    launch_experiments()
