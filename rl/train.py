import torch
import numpy as np
from tqdm import tqdm
import argparse
import random
import os

from agent.Agent import Agent
from agent.memory.ReplayMemory import ReplayMemory
from agent.controllers.DQN import ControllerDQN
from make_plots import show_reward_plot
from logger.Logger import Logger

from envs import CartPole, MountainCar, LunarLander
from params import CartPoleConfig, LunarLanderConfig


logger = Logger("logdir")


def train(episodes, prune_iters, prune_percent, device=torch.device('cpu'), random_state=0):
    env = LunarLander(random_state=random_state)
    config = LunarLanderConfig()
    logger.update_params(config.__dict__)

    memory = ReplayMemory(config.memory_config.memory_size)
    controller = ControllerDQN(env, memory, config, prune_percent=prune_percent, device=device)
    agent = Agent(env, controller, device=device)

    for iter in range(prune_iters):
        plot_data = list()
        pbar = tqdm(range(episodes))
        for episode in pbar:
            pbar.set_description("Iter[{}/{}] Episode [{}/{}]".format(iter + 1, prune_iters, episode + 1, episodes))

            reward, _ = agent.rollout(train=False)
            agent.rollout(train=True)
            
            plot_data.append(reward)
            if controller.optimization_completed() and not iter + 1 == prune_iters: # no stop on last iteration
                break

        show_reward_plot(plot_data)
        torch.save(plot_data, "plots/LunarLander_iter" + str(iter) + "_prune" +
                str(0.8 ** iter))

        
        controller.prune()
        controller.reinit()


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=1000, required=False)
    parser.add_argument('--prune-iters', type=int, default=1, required=False)
    parser.add_argument('--prune-percent', type=float, default=20, required=False)
    parser.add_argument('--device', type=str, default='cpu', required=False)
    return parser


def init_random_seeds(RANDOM_SEED, cuda_determenistic):
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    if cuda_determenistic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    init_random_seeds(179, cuda_determenistic=True)

    args = create_parser().parse_args() 
    logger.update_params(args.__dict__)
    train(args.episodes, args.prune_iters, args.prune_percent, torch.device(args.device), RANDOM_SEED)
