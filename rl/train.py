import torch
from tqdm import tqdm

from agent.Agent import Agent
from agent.ReplayMemory import ReplayMemory
from agent.DQN import ControllerDQN
from envs.CartPole import CartPoleWrapper

device = torch.device("cpu")
env = CartPoleWrapper()

plot_data = list()
agent = Agent(env, ControllerDQN(env.state_sz, env.action_sz, ReplayMemory(10000)), device=device)
epochs=500
pbar = tqdm(range(epochs))
for epoch in pbar:
    reward, steps = agent.rollout(show=False)
    pbar.set_description("Epoch [{}/{}]".format(epoch + 1, epochs))
    pbar.write("Reward: {:.3f}".format(reward))
    plot_data.append((steps, reward))
