import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

HIDDEN_SZ = 256

class ActorNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, HIDDEN_SZ),
            nn.ELU(inplace=True),
            nn.Linear(HIDDEN_SZ, HIDDEN_SZ),
            nn.ELU(inplace=True),
        )
        self.mu_linear = nn.Linear(HIDDEN_SZ, action_dim)
        self.logs_linear = nn.Linear(HIDDEN_SZ, action_dim)

    def forward(self, state):
        x = self.model(state)
        return self.mu_linear(x), self.logs_linear(x)

    def sample(self, state):
        mu, logs = self.forward(state)
        logs = torch.clamp(logs, -20, 2)
        nd = torch.distributions.Normal(mu, logs.exp())
        pure_action = nd.rsample() # reparametrization trick inside
        action = torch.tanh(pure_action)

        log_prob = nd.log_prob(pure_action).sum(dim=1) - torch.log(1 - action ** 2 + 1e-6).sum(dim=1)

        return action, mu, logs, log_prob


class CriticNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim + action_dim, HIDDEN_SZ),
            nn.ELU(inplace=True),
            nn.Linear(HIDDEN_SZ, HIDDEN_SZ),
            nn.ELU(inplace=True),
            nn.Linear(HIDDEN_SZ, 1)
        )

    def forward(self, state, action):
        return self.model(torch.cat([state, action], dim=-1)).squeeze(1)

