from envs.AtariWrapper import AtariWrapper

class Assault(AtariWrapper):
    def __init__(self, random_state):
        super(Assault, self).__init__("Assault-v0", random_state)


