import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import math
import random

from agent.ReplayMemory import Transition
from networks.DQN import DQN
from pruners.LayerwisePruner import LayerwisePruner


class ControllerDQN(nn.Module):
    def __init__(self, env, memory, params, device=torch.device('cpu')):
        super(ControllerDQN, self).__init__()
        self.state_sz = env.state_sz
        self.action_sz = env.action_sz
        self.memory = memory
        self.params = params
        self.device = device

        self.eps_start = params.eps_start
        self.eps_end = params.eps_end
        self.eps_decay = params.eps_decay
        self.batch_size = params.batch_size
        self.gamma = params.gamma
        self.target_net_update_steps = params.target_net_update_steps

        self.net = DQN(self.state_sz, self.action_sz, params.layers_sz).to(device)
        self.target_net = DQN(self.state_sz, self.action_sz, params.layers_sz).to(device)

        self.pruner = LayerwisePruner(self.net, self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=params.optimizer_config.lr)

        self.steps_done = 0

    def select_action(self, state):
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.steps_done / self.eps_decay)
        if random.random() > eps_threshold:
            with torch.no_grad():
                return self.net(state).max(1)[1].item()
        else:
            return random.randrange(self.action_sz)

    def hard_update(self):
        self.target_net.load_state_dict(self.net.state_dict())

    def optimize(self):
        if self.steps_done % self.target_net_update_steps == 0:
            self.hard_update()
        self.steps_done += 1
        if len(self.memory) < self.batch_size:
            return

        state, action, next_state, reward, done = self.memory.sample(self.batch_size)

        state_action_values = self.net(state).gather(1, action.unsqueeze(1))
        with torch.no_grad():
            next_values = self.target_net(next_state).max(1)[0]
            expected_state_action_values = (next_values * self.gamma) * (1 - done) + reward

        loss = F.smooth_l1_loss(state_action_values.squeeze(1), expected_state_action_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.pruner.optimization_step()
        self.optimizer.step()

    def prune(self, p):
        self.pruner.prune_net(p)

    def reinit(self, ):
        self.memory.clean()
        self.steps_done = 0
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.params.optimizer_config.lr)
        self.pruner.reinit_net()
        self.hard_update()

    def save_model(self, path):
        torch.save(self, path)

    def push_in_memory(self, state, action, next_state, reward, done):
        self.memory.push(state, action, next_state, reward, done)

    @staticmethod
    def load_model(path, *args, **kwargs):
        cnt = torch.load(path)
        cnt.to(cnt.device)
        return cnt
