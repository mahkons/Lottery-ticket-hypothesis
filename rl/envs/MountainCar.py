import torch
import gym

from envs.EnvWrapper import EnvWrapper

class MountainCar(EnvWrapper):
    def __init__(self, random_state):
        super(MountainCar, self).__init__("MountainCar-v0", random_state)
