from envs.AtariWrapper import AtariWrapper

class SpaceInvaders(AtariWrapper):
    def __init__(self, random_state):
        super(SpaceInvaders, self).__init__("SpaceInvaders-v0", random_state)



