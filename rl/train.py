import torch
from tqdm import tqdm

from agent.Agent import Agent
from agent.ReplayMemory import ReplayMemory
from agent.DQN import ControllerDQN
from make_plots import show_reward_plot

from envs.CartPole import CartPole
from envs.MountainCar import MountainCar
from envs.LunarLander import LunarLander

device = torch.device("cpu")
env = CartPole()
controller = ControllerDQN(env.state_sz, env.action_sz, ReplayMemory(100000), lr=1e-3, device=device)
agent = Agent(env, controller, device=device)

plot_data = list()
epochs=100
pbar = tqdm(range(epochs))
for epoch in pbar:
    reward, steps = agent.rollout(show=False)
    pbar.set_description("Epoch [{}/{}]".format(epoch + 1, epochs))
    pbar.write("Reward: {:.3f}".format(reward))
    plot_data.append(reward)

show_reward_plot(plot_data)
