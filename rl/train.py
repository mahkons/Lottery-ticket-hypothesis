import torch
import gym
from tqdm import tqdm

from agent.Agent import Agent
from agent.ReplayMemory import ReplayMemory
from agent.DQN import ControllerDQN

device = torch.device("cpu")
env = gym.make("CartPole-v1")
action_sz = env.action_space.n
state_sz = 4

plot_data = list()
agent = Agent(env, ControllerDQN(state_sz, action_sz, ReplayMemory(10000)), device=device)
epochs=1000
pbar = tqdm(range(epochs))
for epoch in pbar:
    reward, steps = agent.rollout(show=False)
    pbar.set_description("Epoch [{}/{}]".format(epoch + 1, epochs))
    pbar.write("Reward: {:.3f}".format(reward))
    plot_data.append((steps, reward))
