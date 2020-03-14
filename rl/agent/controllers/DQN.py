import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import math
import random

from agent.stop_criterions import NoStop, MaskDiffStop, EarlyBirdStop
from agent.memory.ReplayMemory import Transition
from networks.DQN import DQN
from pruners import LayerwisePruner, GlobalPruner, ERPruner, RewindWrapper
from metrics import MetricsDict, Barrier, Metric, DispersionMetric


class ControllerDQN(nn.Module):
    def __init__(self, env, memory, params, prune_percent=20, device=torch.device('cpu')):
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

        self.net = DQN(self.state_sz, self.action_sz, params.layers_sz, params.image_input).to(device)
        self.target_net = DQN(self.state_sz, self.action_sz, params.layers_sz, params.image_input).to(device)

        self.prune_percent = prune_percent
        self.pruner = RewindWrapper(ERPruner(self.net, self.device), 0)
        self.stop_criterion = MaskDiffStop(eps=0)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=params.optimizer_config.lr)

        self.steps_done = 0

        self.metrics = MetricsDict((Metric("qerror"), DispersionMetric("stability", 50)))

        if params.best_model_path != ":(":
            self.best_net = DQN(self.state_sz, self.action_sz, params.layers_sz, params.image_input).to(device)
            self.best_net.load_state_dict(state_dict=torch.load(params.best_model_path))
        else:
            self.best_net = None

    def select_action(self, state, explore):
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.steps_done / self.eps_decay)
        if explore and random.random() < eps_threshold:
            return random.randrange(self.action_sz)
        else:
            with torch.no_grad():
                return self.net(state).max(1)[1].item()

    def hard_update(self):
        self.target_net.load_state_dict(self.net.state_dict())

    def calc_loss(self):
        state, action, next_state, reward, done = self.memory.sample(self.batch_size)

        state_action_values = self.net(state).gather(1, action.unsqueeze(1))
        with torch.no_grad():
            next_values = self.target_net(next_state).max(1)[0]
            expected_state_action_values = (next_values * self.gamma) * (1 - done) + reward

        loss = F.smooth_l1_loss(state_action_values.squeeze(1), expected_state_action_values)

        return loss

    def optimize(self):
        self.steps_done += 1
        if self.steps_done % self.target_net_update_steps == 0:
            self.hard_update()
            self.stop_criterion.update_mask(self.pruner.get_mask_to_prune(self.prune_percent))
            self.pruner.epoch_step()
            self.metrics.add_barrier(Barrier.EPOCH)

        if len(self.memory) < self.batch_size:
            return

        loss = self.calc_loss()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.pruner.optimization_step()

    def optimization_completed(self):
        return self.stop_criterion()

    def prune(self):
        self.metrics.add_barrier(Barrier.PRUNE)
        self.pruner.prune_net(self.prune_percent)

    def reinit(self):
        self.memory.clean()
        self.steps_done = 0
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.params.optimizer_config.lr)
        self.stop_criterion.reset()

        # Next iteration target net will start with NOT cool parameters
        self.pruner.reinit_net()
        self.hard_update()

    def update_metrics(self, state, action, next_state, reward, done):
        if self.best_net:
            state_action_values = self.net(state).gather(1, action.unsqueeze(1))
            best_values = self.best_net(state).gather(1, action.unsqueeze(1))
            self.metrics["qerror"].add(state_action_values.item() - best_values.item())

    def push_in_memory(self, state, action, next_state, reward, done):
        self.update_metrics(state, action, next_state, reward, done)
        self.memory.push(state, action, next_state, reward, done)

    def load_net(self, path):
        self.net.load_state_dict(state_dict=torch.load(path))
        self.hard_update()

    def save_net(self, path):
        torch.save(self.target_net.state_dict(), path)
