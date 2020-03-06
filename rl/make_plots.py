import plotly as plt
import plotly.graph_objects as go
import argparse
from os import walk

import torch
import numpy as np

def add_trace(plot, x, y, name):
    plot.add_trace(go.Scatter(x=x, y=y, name=name))

def add_avg_trace(plot, x, y, name, avg_epochs):
    add_trace(plot, x, make_smooth(y, avg_epochs), name)

def make_smooth(y, avg_epochs=100):
    ny = list()
    cur_val = y[0]
    for i in range(len(y)):
        cur_val -= y[max(0, i - avg_epochs)] / avg_epochs
        cur_val += y[i] / avg_epochs
        ny.append(cur_val)

    return ny


def add_vertical_line(plot, x, y_st, y_en, name):
    add_trace(plot, x=[x, x], y=[y_st, y_en], name=name)


def create_reward_plot(plot_data, title="reward plot", steps=False, avg_epochs=1):
    plot = go.Figure()
    plot.update_layout(title=title)
    train_episodes, steps, rewards = zip(*plot_data)

    y = np.array(rewards)
    if steps:
        x = np.array(steps)
    else:
        x = np.array(train_episodes)
    add_avg_trace(plot, x, y, name="reward", avg_epochs=avg_epochs)

    return plot


def create_metric_plot(plot_data, title="metric", avg_epochs=1, show_epochs=False, show_prune_epochs=True):
    plot = go.Figure()
    plot.update_layout(title=title)

    y_data = list()
    barriers = list()
    for i in range(len(plot_data)):
        if isinstance(plot_data[i], str):
            barriers.append((plot_data[i], len(y_data)))
        else:
            y_data.append(plot_data[i])
            

    y = np.array(make_smooth(y_data, avg_epochs))
    x = np.array(range(len(y)))
    add_trace(plot, x, y, name="metric")

    y_min, y_max = np.min(y), np.max(y)

    for name, pos in barriers:
        if name == "epoch" and not show_epochs:
            continue
        if name == "prune" and not show_prune_epochs:
            continue

        add_vertical_line(plot, pos, y_min, y_max, name=name)

    return plot


if __name__ == "__main__":
    pass
