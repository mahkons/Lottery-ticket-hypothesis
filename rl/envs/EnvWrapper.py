import torch
import gym

class EnvWrapper():
    def __init__(self, name, random_state):
        self.env = gym.make(name)
        self.env.seed(random_state)
        self.action_sz = self.env.action_space.n
        self.state_sz = self.env.observation_space.shape[0]

        self.steps = 0
        self.total_reward = 0

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        real_reward = reward
        self.total_reward += real_reward
        self.steps += 1

        obs = self.transform_obs(obs)
        reward = self.shape_reward(obs, reward, done)

        return obs, reward, done, info, real_reward

    def reset(self):
        self.steps, self.total_reward = 0, 0
        return self.transform_obs(self.env.reset())

    def close(self):
        self.env.close()

    def render(self):
        self.env.render()

    def get_steps(self):
        return self.steps

    def get_total_reward(self):
        return self.total_reward

    def shape_reward(self, obs, reward, done):
        return reward

    def transform_obs(self, obs):
        return obs

    def sample_action(self):
        return self.env.action_space.sample()
