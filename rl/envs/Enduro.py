from envs.AtariWrapper import AtariWrapper

class Enduro(AtariWrapper):
    def __init__(self, random_state):
        super(Enduro, self).__init__("Enduro-v0", random_state)



