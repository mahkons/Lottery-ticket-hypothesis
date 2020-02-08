import torch
import gym

from envs.EnvWrapper import EnvWrapper

class CartPole(EnvWrapper):
    def __init__(self):
        super(CartPole, self).__init__("CartPole-v1")
