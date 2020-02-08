import numpy as np
from collections import namedtuple
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F

class Agent:
    def __init__(self, env, controller, device):
        self.env = env
        self.controller = controller
        self.device = device

    def wrap(self, x, dtype=torch.float):
        return torch.tensor([x], dtype=dtype, device=self.device)

    def rollout(self, show=False):
        state = self.wrap(self.env.reset())

        while True:
            if show:
                self.env.render()

            action = self.controller.select_action(state)
            obs, reward, done, _, _ = self.env.step(action)

            reward = self.wrap(reward)
            action = self.wrap(action, dtype=torch.long)
            next_state = self.wrap(obs)

            self.controller.push_in_memory(state, action, next_state, reward, self.wrap(done))
            state = next_state

            self.controller.optimize()
            if done:
                break;
        return self.env.total_reward, self.env.steps
