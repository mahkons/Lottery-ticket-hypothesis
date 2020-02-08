import torch
import gym

from envs.EnvWrapper import EnvWrapper

class LunarLander(EnvWrapper):
    def __init__(self):
        super(LunarLander, self).__init__("LunarLander-v2")
