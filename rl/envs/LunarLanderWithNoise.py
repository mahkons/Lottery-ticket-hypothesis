import numpy as np

from envs.EnvWrapper import EnvWrapper

class LunarLanderWithNoise(EnvWrapper):
    def __init__(self, random_state):
        super(LunarLanderWithNoise, self).__init__("LunarLander-v2", random_state)
        self.state_sz = 256

    def transform_obs(self, obs):
        return np.concatenate((obs, np.random.uniform(size=248)))
