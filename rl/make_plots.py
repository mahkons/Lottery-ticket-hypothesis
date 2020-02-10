import plotly as plt
import plotly.graph_objects as go
import argparse
from os import walk

import torch
import numpy as np

def add_trace(plot, x, y, name):
    plot.add_trace(go.Scatter(x=x, y=y, name=name))

def add_avg_trace(plot, x, y, name, avg_epochs=100):
    ny = list()
    cur_val = y[0]
    for i in range(len(y)):
        cur_val -= y[max(0, i - avg_epochs)] / avg_epochs
        cur_val += y[i] / avg_epochs
        ny.append(cur_val)

    add_trace(plot, x, ny, name)

def show_reward_plot(plot_data, name="reward plot", avg_epochs=100):
    plot = go.Figure()

    y = np.array(plot_data)
    x = np.arange(len(y))
    add_avg_trace(plot, x, y, name, avg_epochs)

    plot.show()


if __name__ == "__main__":
    plot = go.Figure()

    paths = list()
    for _, _, files in walk("plots/"):
        paths += files

    for path in paths:
        plot_data = torch.load("plots/" + path)
        add_avg_trace(plot, np.arange(len(plot_data)), plot_data, path)

    plot.show()
