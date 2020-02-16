from make_plots import show_reward_plot


class MaskDiffStop():
    def __init__(self, eps=0.1):
        self.mask_dict = None
        self.plot_data = list()

    def update_mask(self, next_mask_dict):
        if self.mask_dict is None:
            self.mask_dict = next_mask_dict
            return

        sum_diff = 0
        for key in self.mask_dict:
            sum_diff += (self.mask_dict[key] ^ next_mask_dict[key]).sum()
        self.plot_data.append(sum_diff.item())
        self.mask_dict = next_mask_dict

    def __call__(self):
        return False

    def reset(self):
        self.mask_dict = None
