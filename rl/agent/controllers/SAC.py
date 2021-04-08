import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import itertools
import copy

from networks import ActorNet, CriticNet
from ReplayBuffer import ReplayBuffer
from params import DEVICE, BUFFER_SIZE, INIT_BUFFER_SIZE, LR, BATCH_SIZE, GAMMA, TAU


class SAC():
    def __init__(self, state_dim, action_dim):
        self.actor = ActorNet(state_dim, action_dim).to(DEVICE)
        self.critic_1 = CriticNet(state_dim, action_dim).to(DEVICE)
        self.critic_2 = CriticNet(state_dim, action_dim).to(DEVICE)

        self.target_entropy = -action_dim
        self.log_alpha = torch.tensor(0., dtype=torch.float, requires_grad=True, device=DEVICE)
        self.alpha = self.log_alpha.exp()

        self.critic_optimizer = torch.optim.Adam(itertools.chain(self.critic_1.parameters(), self.critic_2.parameters()), lr=LR)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=LR)
        self.alpha_optimizer = torch.optim.Adam((self.log_alpha,), lr=LR)

        self.target_critic_1 = CriticNet(state_dim, action_dim).to(DEVICE)
        self.target_critic_2 = CriticNet(state_dim, action_dim).to(DEVICE)
        self.hard_update()

        self.replay_buffer = ReplayBuffer(capacity=BUFFER_SIZE)


    def act(self, state, train):
        with torch.no_grad():
            state = torch.tensor(np.array([state]), dtype=torch.float, device=DEVICE)
            action, mu, logs, logprob = self.actor.sample(state)
            if train:
                return action[0].cpu().numpy()
            else:
                return torch.tanh(mu)[0].cpu().numpy()

    def update(self, transition):
        self.replay_buffer.push(*transition)

    def optimize(self):
        if len(self.replay_buffer) < INIT_BUFFER_SIZE:
            return
        state, action, next_state, reward, done = self.replay_buffer.sample(BATCH_SIZE, DEVICE)

        with torch.no_grad():
            next_action, _, _, na_logprob = self.actor.sample(next_state)
            q1_target, q2_target = self.target_critic_1(next_state, next_action), self.target_critic_2(next_state, next_action)
            min_q_target = reward + GAMMA * (1 - done) * (torch.min(q1_target, q2_target) - self.alpha * na_logprob)

        q1_value, q2_value = self.critic_1(state, action), self.critic_2(state, action)
        q1_loss = F.mse_loss(q1_value, min_q_target)
        q2_loss = F.mse_loss(q2_value, min_q_target)

        self.critic_optimizer.zero_grad()
        (q1_loss + q2_loss).backward()
        self.critic_optimizer.step()

        cur_action, _, _, cur_logprob = self.actor.sample(state)
        cur_q1_value, cur_q2_value = self.critic_1(state, cur_action), self.critic_2(state, cur_action)

        actor_loss = (self.alpha.detach() * cur_logprob - torch.min(cur_q1_value, cur_q2_value)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = -self.log_alpha * (self.target_entropy + cur_logprob.detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp()

        self.soft_update()

    def hard_update(self):
        _hard_update(self.target_critic_1, self.critic_1)
        _hard_update(self.target_critic_2, self.critic_2)

    def soft_update(self):
        _soft_update(self.target_critic_1, self.critic_1, TAU)
        _soft_update(self.target_critic_2, self.critic_2, TAU)

    def save(self):
        torch.save(self.actor, "agent.pkl")


def _soft_update(target, source, tau):
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_((1 - tau) * tp.data + tau * sp.data)

def _hard_update(target, source):
    target.load_state_dict(source.state_dict())
