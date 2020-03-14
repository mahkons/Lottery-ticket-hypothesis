import torch
import numpy as np
from tqdm import tqdm
import argparse
import random
import os
import datetime

from agent.Agent import Agent
from agent.memory.ReplayMemory import ReplayMemory
from agent.controllers.DQN import ControllerDQN
from make_plots import create_reward_plot, create_metric_plot
from logger.Logger import log, init_logger

from envs import CartPole, LunarLander, Pong, Breakout
from params import CartPoleConfig, LunarLanderConfig, AtariConfig, BigLunarLanderConfig


def explore(agent, train_episode, plot_name):
    reward, steps = agent.rollout(train=True)
    log().add_plot_point(plot_name, (train_episode, agent.controller.steps_done, reward))


def exploit(agent, train_episode, plot_name):
    reward, steps = agent.rollout(train=False)
    log().add_plot_point(plot_name, (train_episode, agent.controller.steps_done, reward))
    agent.controller.metrics["stability"].add(reward)


def train(episodes, opt_steps, prune_iters, prune_percent, device, random_state):
    env = LunarLander(random_state=random_state)
    config = LunarLanderConfig()
    log().update_params(config.to_dict())

    memory = ReplayMemory(config.memory_config.memory_size)
    controller = ControllerDQN(env, memory, config, prune_percent=prune_percent, device=device)
    agent = Agent(env, controller, device=device)

    EXPLORE_ITERS = 1
    EXPLOIT_ITERS = 1

    for iter in range(prune_iters):
        pbar = tqdm(range(episodes))
        cur_percent = (1 - prune_percent / 100) ** iter
        explore_plot = "Explore_iter" + str(iter) + "_prune" + str(cur_percent)
        exploit_plot = "Exploit_iter" + str(iter) + "_prune" + str(cur_percent)
        log().add_plot(explore_plot, columns=("train_episode", "train_steps", "reward"))
        log().add_plot(exploit_plot, columns=("train_episode", "train_steps", "reward"))

        for episode in pbar:
            # once in EXPLORE_ITERS train rollouts, do EXPLOIT_ITERS exploit rollouts
            if episode % EXPLORE_ITERS == EXPLORE_ITERS - 1:
                for _ in range(EXPLOIT_ITERS):
                    pbar.set_description("Iter[{}/{}] Episode [{}/{}] Step[{}/{}] Exploit"
                            .format(iter + 1, prune_iters, episode + 1, episodes, controller.steps_done, opt_steps))
                    exploit(agent, episode, exploit_plot)

            pbar.set_description("Iter[{}/{}] Episode [{}/{}] Step[{}/{}] Explore"
                    .format(iter + 1, prune_iters, episode + 1, episodes, controller.steps_done, opt_steps))
            explore(agent, episode, explore_plot)

            if controller.steps_done >= opt_steps:
                break
            if controller.optimization_completed() and not iter + 1 == prune_iters: # no stop on last iteration
                break

        controller.prune()
        controller.reinit()


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prune-iters', type=int, default=1, required=False)
    parser.add_argument('--prune-percent', type=float, default=20, required=False)
    parser.add_argument('--device', type=str, default='cpu', required=False)

    parser.add_argument('--logname', type=str, default="log_" + datetime.datetime.now().isoformat(), required=False)

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--episodes', type=int, default=10**10)
    group.add_argument('--opt-steps', type=int, default=10**10)
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
    RANDOM_SEED = 9
    init_random_seeds(RANDOM_SEED, cuda_determenistic=True)

    args = create_parser().parse_args() 

    init_logger("logdir", args.logname)
    log().update_params(args.__dict__)

    try:
        train(args.episodes, args.opt_steps,
                args.prune_iters, args.prune_percent, torch.device(args.device), RANDOM_SEED)
    finally:
        log().save_logs()

