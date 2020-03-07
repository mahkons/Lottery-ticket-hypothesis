import torch
from metrics.Metric import Metric
from metrics.Barrier import Barrier

class MetricsDict():
    def __init__(self, metrics_names):
        self.metrics_names = metrics_names
        self.dict = dict()
        for metric in metrics_names:
            self.dict[metric] = Metric(metric)

    def add_barrier(self, barrier: Barrier):
        for name, data in self.dict.items():
            data.add(int(barrier)) # int!

    def __getitem__(self, metric):
        return self.dict[metric]
