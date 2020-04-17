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

def load_experiments(exp_path, sdevice):
    exp_list = torch.load(exp_path)
    
    # TODO a better way to choose cuda device for experiment
    # by the way only 4 experiments and 4 cuda is common case
    count = 0
    for exp in exp_list:
        if sdevice == "cpu":
            device = torch.device("cpu")
        elif sdevice == "cuda":
            device = torch.device("cuda:" + str(count % torch.cuda.device_count()))
        else:
            assert False

        print(device)
        exp.device = device

        count += 1

    return exp_list


def launch_experiments(exp_list, processes=4):
    with Pool(processes=processes) as pool:
        pool.map(start_experiment, exp_list)


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--processes', type=int, default=4, required=False)
    parser.add_argument('--device', type=str, default="cpu", required=False)
    parser.add_argument('--exp-path', type=str, default="generated/exp_list", required=False)
    return parser


if __name__ == "__main__":
    args = create_parser().parse_args()
    launch_experiments(load_experiments(args.exp_path, args.device), args.processes)
