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

from configs import Experiment
from envs import CartPole, LunarLander, Pong, Breakout
from params import CartPoleConfig, LunarLanderConfig, AtariConfig, BigLunarLanderConfig
from pruners import LayerwisePruner, GlobalPruner, ERPruner, RewindWrapper
from agent.stop_criterions import NoStop, MaskDiffStop, EarlyBirdStop


def explore(agent, train_episode, plot_name):
    reward, steps = agent.rollout(train=True)
    log().add_plot_point(plot_name, (train_episode, agent.controller.steps_done, reward))


def exploit(agent, train_episode, plot_name):
    reward, steps = agent.rollout(train=False)
    log().add_plot_point(plot_name, (train_episode, agent.controller.steps_done, reward))
    agent.controller.metrics["stability"].add(reward)


def train(experiment):
    env = experiment.env(random_state=experiment.random_seed)
    memory = ReplayMemory(experiment.hyperparams.memory_config.memory_size)
    controller = ControllerDQN(
            env = env,
            memory = memory,
            params = experiment.hyperparams,
            prune_percent = experiment.prune_percent,
            pruner = experiment.pruner,
            stop_criterion = experiment.stop_criterion,
            device = experiment.device
        )
    agent = Agent(env, controller, device=experiment.device)

    EXPLORE_ITERS = 1
    EXPLOIT_ITERS = 1

    episodes, prune_iters, opt_steps = experiment.episodes, experiment.prune_iters, experiment.opt_steps

    for iter in range(prune_iters):
        pbar = tqdm(range(episodes))
        cur_percent = (1 - experiment.prune_percent / 100) ** iter
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

            torch.cuda.empty_cache()

        controller.prune()
        controller.reinit()

        log().save_logs()


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


def start_experiment(experiment):
    init_random_seeds(experiment.random_seed, cuda_determenistic=True)
    init_logger("logdir", experiment.logname)
    log().update_params(experiment.to_dict())

    try:
        train(experiment)
    finally:
        log().save_logs()


if __name__ == "__main__":
    args = create_parser().parse_args() 

    RANDOM_SEED = 2020
    device = torch.device(args.device)

    experiment = Experiment(
            opt_steps = args.opt_steps,
            episodes = args.episodes,
            prune_iters = args.prune_iters,
            prune_percent = args.prune_percent,
            device = device,
            logname = args.logname,
            random_seed = RANDOM_SEED,
            env = LunarLander,
            hyperparams = LunarLanderConfig(),
            stop_criterion = MaskDiffStop(eps=0),
            pruner = lambda net: RewindWrapper(ERPruner(net, device), 0, rescale=False),
        )

    start_experiment(experiment)
