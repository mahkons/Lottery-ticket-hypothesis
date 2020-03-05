import torch
from metrics.Metric import Metric

class MetricsDict():
    def __init__(self, metrics_names):
        self.metrics_names = metrics_names
        self.dict = dict()

    def __getitem__(self, metric):
        assert metric in self.metrics_names
        
        if not metric in self.dict:
            self.dict[metric] = Metric(metric)
        return self.dict[metric]
