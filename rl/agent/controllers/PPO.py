import torch
import numpy as np
import itertools

from networks import Actor, Critic
from params import LR, GAMMA, LAMBDA, CLIP, ENTROPY_COEF, VALUE_COEFF, BATCHES_PER_UPDATE, BATCH_SIZE


class PPO():
    def __init__(self, state_dim, action_dim):
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim)
        self.optimizer = torch.optim.Adam(itertools.chain(self.actor.parameters(), self.critic.parameters()), LR)

    def _calc_loss(self, state, action, old_log_prob, expected_values, gae):
        new_log_prob, action_distr = self.actor.compute_proba(state, action)
        state_values = self.critic.get_value(state).squeeze(1)

        critic_loss = ((expected_values - state_values) ** 2).mean()

        unclipped_ratio = torch.exp(new_log_prob - old_log_prob)
        clipped_ratio = torch.clamp(unclipped_ratio, 1 - CLIP, 1 + CLIP)
        actor_loss = -torch.min(clipped_ratio * gae, unclipped_ratio * gae).mean()

        entropy_loss = -action_distr.entropy().mean()

        return critic_loss * VALUE_COEFF + actor_loss + entropy_loss * ENTROPY_COEF


    def update(self, trajectories):
        trajectories = map(self._compute_lambda_returns_and_gae, trajectories)
        transitions = sum(trajectories, []) # Turn a list of trajectories into list of transitions

        state, action, old_log_prob, target_value, advantage = zip(*transitions)
        state = np.array(state)
        action = np.array(action)
        old_log_prob = np.array(old_log_prob)
        target_value = np.array(target_value)
        advantage = np.array(advantage)
        advnatage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        
        for _ in range(BATCHES_PER_UPDATE):
            idx = np.random.randint(0, len(transitions), BATCH_SIZE) # Choose random batch
            s = torch.from_numpy(state[idx]).float()
            a = torch.from_numpy(action[idx]).float()
            op = torch.from_numpy(old_log_prob[idx]).float() # Log probability of the action in state s.t. old policy
            v = torch.from_numpy(target_value[idx]).float() # Estimated by lambda-returns 
            adv = torch.from_numpy(advantage[idx]).float() # Estimated by generalized advantage estimation 

            loss = self._calc_loss(s, a, op, v, adv)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


    def _compute_lambda_returns_and_gae(self, trajectory):
        lambda_returns = []
        gae = []
        last_lr = 0.
        last_v = 0.
        for s, _, r, _ in reversed(trajectory):
            ret = r + GAMMA * (last_v * (1 - LAMBDA) + last_lr * LAMBDA)
            last_lr = ret
            last_v = self.get_value(s)
            lambda_returns.append(last_lr)
            gae.append(last_lr - last_v)
        
        # Each transition contains state, action, old action probability, value estimation and advantage estimation
        return [(s, a, p, v, adv) for (s, a, _, p), v, adv in zip(trajectory, reversed(lambda_returns), reversed(gae))]
            
            
    def get_value(self, state):
        with torch.no_grad():
            state = torch.from_numpy(state).float().unsqueeze(0)
            value = self.critic.get_value(state)
        return value.cpu().item()

    def act(self, state):
        with torch.no_grad():
            state = torch.from_numpy(state).float().unsqueeze(0)
            action, pure_action, log_prob = self.actor.act(state)
        return action.cpu().numpy()[0], pure_action.cpu().numpy()[0], log_prob.cpu().item()

    def save(self):
        torch.save(self.actor, "agent.pkl")


