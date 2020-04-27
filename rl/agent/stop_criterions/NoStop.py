from .StopCriterion import StopCriterion

class NoStop(StopCriterion):
    def update_mask(self, next_mask_dict):
        pass

    def __call__(self, *args, **kwargs):
        return False

    def reset(self):
        pass
