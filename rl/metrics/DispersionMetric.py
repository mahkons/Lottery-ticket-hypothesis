import torch
from collections import deque

from logger.Logger import log
from metrics.Metric import Metric

class DispersionMetric(Metric):
    def __init__(self, name, window_length):
        super(DispersionMetric, self).__init__(name)

        self.window_length = window_length
        self.window_sum = 0
        self.window_square_sum = 0
        self.window = deque()

    def add(self, value):
        if len(self.window) == self.window_length:
            x = self.window.popleft()
            self.window_sum -= x
            self.window_square_sum -= x ** 2

        self.window.append(value)
        self.window_sum += value
        self.window_square_sum += value ** 2

        dispersion = self.window_square_sum / len(self.window)  - (self.window_sum / len(self.window)) ** 2
        super().add(dispersion)

