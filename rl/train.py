import torch
import numpy as np
from tqdm import tqdm
import argparse
import random

from agent.Agent import Agent
from agent.memory.ReplayMemory import ReplayMemory
from agent.controllers.DQN import ControllerDQN
from make_plots import show_reward_plot

from envs.CartPole import CartPole
from envs.MountainCar import MountainCar
from envs.LunarLander import LunarLander

from params import CartPoleConfig, LunarLanderConfig


def train(epochs, prune_iters, device=torch.device('cpu'), random_state=0):
    env = LunarLander(random_state=random_state)
    config = LunarLanderConfig()

    memory = ReplayMemory(config.memory_config.memory_size)
    controller = ControllerDQN(env, memory, config, device=device)
    agent = Agent(env, controller, device=device)

    for iter in range(prune_iters):
        plot_data = list()
        pbar = tqdm(range(epochs))
        for epoch in pbar:
            reward, steps = agent.rollout(show=False)
            pbar.set_description("Iter[{}/{}] Epoch [{}/{}]".format(iter + 1, prune_iters, epoch + 1, epochs))
            pbar.write("Reward: {:.3f}".format(reward))
            plot_data.append(reward)
            if controller.optimization_completed():
                break
        show_reward_plot(controller.stop_criterion.plot_data, avg_epochs=1)

        show_reward_plot(plot_data)
        torch.save(plot_data, "plots/LunarLander_iter" + str(iter) + "_prune" +
                str(0.8 ** iter))

        controller.prune(20)
        controller.reinit()


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1000, required=False)
    parser.add_argument('--prune-iters', type=int, default=1, required=False)
    parser.add_argument('--device', type=str, default='cpu', required=False)
    return parser


if __name__ == "__main__":
    RANDOM_SEED = 179
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    args = create_parser().parse_args() 
    train(args.epochs, args.prune_iters, torch.device(args.device), RANDOM_SEED)
