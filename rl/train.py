import torch
from tqdm import tqdm

from agent.Agent import Agent
from agent.ReplayMemory import ReplayMemory
from agent.DQN import ControllerDQN
from make_plots import show_reward_plot

from envs.CartPole import CartPole
from envs.MountainCar import MountainCar
from envs.LunarLander import LunarLander

from params import CartPoleConfig, LunarLanderConfig

device = torch.device("cpu")

env = LunarLander()
config = LunarLanderConfig()

memory = ReplayMemory(config.memory_config.memory_size)
controller = ControllerDQN(env, memory, config, device=device)
agent = Agent(env, controller, device=device)

for iter in range(15):
    plot_data = list()
    epochs=1000
    pbar = tqdm(range(epochs))
    for epoch in pbar:
        reward, steps = agent.rollout(show=False)
        pbar.set_description("Epoch [{}/{}]".format(epoch + 1, epochs))
        pbar.write("Reward: {:.3f}".format(reward))
        plot_data.append(reward)

    show_reward_plot(plot_data)
    torch.save(plot_data, "plots/LunarLander_iter" + str(iter) + "_prune" + str(0.8 ** iter))

    controller.prune(20)
    controller.reinit()
