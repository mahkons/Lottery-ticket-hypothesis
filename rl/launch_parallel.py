import torch
from multiprocessing import Pool
from functools import partial
import argparse
from copy import deepcopy

from train import start_experiment
from configs import Experiment
from envs import LunarLander
from params import LunarLanderConfig
from pruners import RewindWrapper, ERPruner, LayerwisePruner, GlobalPruner
from agent.stop_criterions import MaskDiffStop, EarlyBirdStop, NoStop

device = None

def make_pruner__(rewind_epoch, rescale, reinit_to_random, pruner_constructor, net):
    return RewindWrapper(pruner_constructor(net, device), rewind_epoch, rescale=rescale, reinit_to_random=reinit_to_random)

def make_pruner(rewind_epoch, rescale, reinit_to_random, pruner_constructor):
    return partial(make_pruner__, rewind_epoch, rescale, reinit_to_random, pruner_constructor)

def load_experiments(exp_path, device):
    exp_list = torch.load(exp_path)
    for exp in exp_list:
        exp.device = device

    return exp_list


def launch_experiments(exp_list, processes=4):
    with Pool(processes=processes) as pool:
        pool.map(start_experiment, exp_list)


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--processes', type=int, default=4, required=False)
    parser.add_argument('--device', type=str, default='cpu', required=False)
    parser.add_argument('--exp-path', type=str, required=True)
    return parser


if __name__ == "__main__":
    args = create_parser().parse_args()
    device = torch.device(args.device)
    launch_experiments(load_experiments(args.exp_path, device), args.processes)
