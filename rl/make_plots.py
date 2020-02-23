import plotly as plt
import plotly.graph_objects as go
import argparse
from os import walk

import torch
import numpy as np

def add_trace(plot, x, y, name):
    plot.add_trace(go.Scatter(x=x, y=y, name=name))

def add_avg_trace(plot, x, y, name="", avg_epochs=100):
    ny = list()
    cur_val = y[0]
    for i in range(len(y)):
        cur_val -= y[max(0, i - avg_epochs)] / avg_epochs
        cur_val += y[i] / avg_epochs
        ny.append(cur_val)

    add_trace(plot, x, ny, name)

def create_reward_plot(plot_data, title="reward-episode plot", avg_epochs=1):
    plot = go.Figure()
    plot.update_layout(title=title)
    train_episodes, steps, rewards = zip(*plot_data)

    y = np.array(rewards)
    x = np.array(train_episodes)
    add_avg_trace(plot, x, y, avg_epochs)

    return plot

def create_reward_steps_plot(plot_data, title="reward-steps plot", avg_epochs=1):
    plot = go.Figure()
    plot.update_layout(title=title)
    train_episodes, steps, rewards = zip(*plot_data)

    y = np.array(rewards)
    x = np.array(steps)
    add_avg_trace(plot, x, y, avg_epochs)

    return plot


if __name__ == "__main__":
    pass
