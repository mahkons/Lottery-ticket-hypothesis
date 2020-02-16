from collections import deque
from itertools import combinations


class EarlyBirdStop():
    def __init__(self, eps=0.01, queue_size=5):
        self.mask_queue = deque(maxlen=queue_size)
        self.stop = False
        self.eps = eps

    def calc_diff(self, mask1, mask2):
        sum_diff = 0
        for key in mask1:
            sum_diff += (mask1[key] ^ mask2[key]).sum()
        return sum_diff.item()

    def update_mask(self, next_mask_dict):
        self.mask_queue.append(next_mask_dict)
        cur_sum = sum(map(lambda x: x.sum().item(), self.mask_queue[0].values()))

        for mask1, mask2 in combinations(self.mask_queue, 2):
            sum_diff = self.calc_diff(mask1, mask2)
            if sum_diff < self.eps * cur_sum:
                self.stop = True

    def __call__(self):
        return self.stop

    def reset(self):
        self.mask_queue.clear()
        self.stop = False

