from make_plots import show_reward_plot

class MaskDiffStop():
    def __init__(self, eps=0.02):
        self.mask_dict = None
        self.stop = False
        self.eps = eps

    def update_mask(self, next_mask_dict):
        if self.mask_dict is None:
            self.mask_dict = next_mask_dict
            return

        sum_diff = 0
        for key in self.mask_dict:
            sum_diff += (self.mask_dict[key] ^ next_mask_dict[key]).sum()
        self.mask_dict = next_mask_dict

        cur_sum = sum(map(lambda x: x.sum().item(), self.mask_dict.values()))
        if sum_diff < self.eps * cur_sum:
            self.stop = True

    def __call__(self):
        return self.stop

    def reset(self):
        self.mask_dict = None
        self.stop = False
