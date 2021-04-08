import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 400),
            nn.ELU(inplace=True),
            nn.Linear(400, 300),
            nn.ELU(inplace=True),
            nn.Linear(300, 150),
            nn.ELU(inplace=True),
            nn.Linear(150, action_dim)
        )
        self.sigma = nn.Parameter(torch.zeros(action_dim))
        
    def compute_proba(self, state, action):
        mu = self.model(state)
        sigma = torch.exp(self.sigma).unsqueeze(0).expand_as(mu)
        distr = torch.distributions.Normal(mu, sigma)
        return distr.log_prob(action).sum(axis=1), distr
        
    def act(self, state):
        mu = self.model(state)
        sigma = torch.exp(self.sigma).unsqueeze(0).expand_as(mu)
        distr = torch.distributions.Normal(mu, sigma)
        pure_action = distr.sample()
        action = torch.tanh(pure_action)
        return action, pure_action, distr.log_prob(pure_action).sum(axis=1)
        
        
class Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 400),
            nn.ELU(inplace=True),
            nn.Linear(400, 300),
            nn.ELU(inplace=True),
            nn.Linear(300, 150),
            nn.ELU(inplace=True),
            nn.Linear(150, 1)
        )
        
    def get_value(self, state):
        return self.model(state)

