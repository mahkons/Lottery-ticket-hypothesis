import argparse
import pandas as pd
from tqdm import tqdm
import os
import torch
import json

from agent.Agent import Agent
from envs import LunarLander, Assault
from agent.controllers.FixedController import FixedController
from networks.DQN import DQN
from logger.Logger import log, init_logger


def launch_after_training(params, net_state_dict, device, episodes, opt_steps):
    env = Assault(23)
    net = DQN(env.state_sz, env.action_sz, "vae", params["image_input"] == "True", device=device).to(device)
    net.load_state_dict(net_state_dict)
    controller = FixedController(lambda state, explore: net(state.to(device)).max(1)[1].item())
    agent = Agent(env, controller)

    plot_name = "AfterTraining"
    log().add_plot(plot_name, columns=("train_episode", "train_steps", "reward"))
    pbar = tqdm(range(episodes))
    total_steps = 0
    for episode in pbar:
        pbar.set_description("Episode [{}/{}] Step[{}/{}] Exploit"
                .format(episode + 1, episodes, total_steps, opt_steps))

        reward, steps = agent.rollout(train=False)
        total_steps += steps
        log().add_plot_point(plot_name, (episode, total_steps, reward))

        if total_steps >= opt_steps:
            break

    log().save_logs()


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu', required=False)
    parser.add_argument('--logname', type=str, required=True)
    parser.add_argument('--oldpath', type=str, required=True)
    parser.add_argument('--modeliter', type=int, required=True)

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--episodes', type=int, default=10**10)
    group.add_argument('--opt-steps', type=int, default=10**10)
    return parser


if __name__ == "__main__":
    args = create_parser().parse_args() 
    init_logger("logdir", args.logname)
    logpath = os.path.join("logdir", args.oldpath)
    device = torch.device(args.device)

    params_path = os.path.join(logpath, "params.csv")
    params = pd.read_csv(params_path, index_col="name").to_dict()["value"]

    model_path = list(filter(lambda path: path.startswith("model:iter{}:".format(args.modeliter)),
        os.listdir(os.path.join(logpath, "models"))))
    assert(len(model_path) == 1)
    model_path = os.path.join(logpath, "models", model_path[0])
    net_state_dict = torch.load(model_path, map_location=device)

    launch_after_training(params, net_state_dict, device, args.episodes, args.opt_steps)
