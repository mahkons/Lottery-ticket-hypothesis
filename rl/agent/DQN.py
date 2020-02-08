import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import math
import random

from agent.ReplayMemory import Transition
from networks.DQN import DQN

BATCH_SIZE = 64
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TARGET_UPDATE = 500


class ControllerDQN(nn.Module):
    def __init__(self, state_sz, action_sz, memory, lr=1e-3, device=torch.device('cpu')):
        super(ControllerDQN, self).__init__()
        self.state_sz = state_sz
        self.action_sz = action_sz
        self.memory = memory
        self.device = device

        self.net = DQN(state_sz, action_sz, [256]).to(device)
        self.target_net = DQN(state_sz, action_sz, [256]).to(device)

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)

        self.steps_done = 0

    def select_action(self, state):
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * self.steps_done / EPS_DECAY)
        if random.random() > eps_threshold:
            with torch.no_grad():
                return self.net(state).max(1)[1].item()
        else:
            return random.randrange(self.action_sz)

    def hard_update(self):
        self.target_net.load_state_dict(self.net.state_dict())

    def optimize(self):
        if self.steps_done % TARGET_UPDATE == 0:
            self.hard_update()
        self.steps_done += 1
        if len(self.memory) < BATCH_SIZE:
            return

        state, action, next_state, reward, done = self.memory.sample(BATCH_SIZE)

        state_action_values = self.net(state).gather(1, action.unsqueeze(1))
        with torch.no_grad():
            next_values = self.target_net(next_state).max(1)[0]
            expected_state_action_values = (next_values * GAMMA) + reward * (1 - done)

        loss = F.smooth_l1_loss(state_action_values.squeeze(1), expected_state_action_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_model(self, path):
        torch.save(self, path)

    def push_in_memory(self, state, action, next_state, reward, done):
        self.memory.push(state, action, next_state, reward, done)

    @staticmethod
    def load_model(path, *args, **kwargs):
        cnt = torch.load(path)
        cnt.to(cnt.device)
        return cnt
