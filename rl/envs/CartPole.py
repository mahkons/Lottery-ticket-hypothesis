from envs.EnvWrapper import EnvWrapper

class CartPole(EnvWrapper):
    def __init__(self, random_state):
        super(CartPole, self).__init__("CartPole-v1", random_state)
