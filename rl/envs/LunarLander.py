from envs.EnvWrapper import EnvWrapper

class LunarLander(EnvWrapper):
    def __init__(self, random_state):
        super(LunarLander, self).__init__("LunarLander-v2", random_state)
