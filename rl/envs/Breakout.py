from envs.AtariWrapper import AtariWrapper

class Breakout(AtariWrapper):
    def __init__(self, random_state):
        super(Breakout, self).__init__("Breakout-v0", random_state)


