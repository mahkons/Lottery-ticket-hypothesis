import torch

class Metric():
    def __init__(self, name):
        self.data = list()
        self.name = name

    def add(self, value):
        self.data.append(value)
