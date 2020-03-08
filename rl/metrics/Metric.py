import torch
from logger.Logger import log

# uses global logger if available
class Metric():
    def __init__(self, name):
        self.name = name
        log().add_plot(name, columns=("metric_value",))

    def add(self, value):
        self.add__(value)

    def add_barrier(self, value):
        self.add__(value)

    def add__(self, value):
        log().add_plot_point(self.name, value)

    def get_plot(self):
        return log().get_plot(self.name)
