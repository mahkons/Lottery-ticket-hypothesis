from envs.AtariWrapper import AtariWrapper

class Pong(AtariWrapper):
    def __init__(self, random_state):
        super(Pong, self).__init__("Pong-v0", random_state)

