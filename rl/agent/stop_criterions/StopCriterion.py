
# interface for stop criterions
class StopCriterion():
    def update_mask(self, next_mask_dict):
        raise NotImplementedError()

    def __call__(self):
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()
