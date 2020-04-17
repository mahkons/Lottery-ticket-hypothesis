from envs.AtariWrapper import AtariWrapper

class RoadRunner(AtariWrapper):
    def __init__(self, random_state):
        super(RoadRunner, self).__init__("RoadRunner-v0", random_state)


