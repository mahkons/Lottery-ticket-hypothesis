import torch
from metrics.Metric import Metric
from metrics.Barrier import Barrier

class MetricsDict():
    def __init__(self, metrics):
        self.dict = dict()
        for metric in metrics:
            assert not metric.name in self.dict
            self.dict[metric.name] = metric

    def add_barrier(self, barrier: Barrier):
        for name, data in self.dict.items():
            data.add_barrier(int(barrier)) # int!

    def __getitem__(self, metric):
        return self.dict[metric]
