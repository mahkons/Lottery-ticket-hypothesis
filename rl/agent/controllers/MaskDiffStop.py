

class MaskDiffStop():
    def __init__(self, init_mask, eps=0.1):
        self.mask_dict = init_mask

    def __call__(self, next_mask_dict):
        sum_diff = 0
        for key in self.mask_dict:
            sum_diff += (self.mask_dict[key] ^ next_mask_dict[key]).sum()
        print(sum_diff)
        self.mask_dict = next_mask_dict
        return False
