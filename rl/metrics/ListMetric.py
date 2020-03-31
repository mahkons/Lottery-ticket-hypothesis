import torch
import json

from logger.Logger import log
from metrics.Metric import Metric

# save lists using torch
class ListMetric(Metric):
    def add(self, value):
        log().add_plot_point(self.name, json.dumps([x.item() for x in value]))

    def add_barrier(self, value):
        pass
