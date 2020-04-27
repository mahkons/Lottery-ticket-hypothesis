from .StopCriterion import StopCriterion

class FixedEpochsStop(StopCriterion):
    def __init__(self, limit_epochs):
        self.limit_epochs = limit_epochs
        self.epochs = 0

    def update_mask(self, next_mask_dict):
        self.epochs += 1

    def __call__(self):
        return self.epochs >= self.limit_epochs

    def reset(self):
        self.epochs = 0
