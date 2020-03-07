import plotly as plt
import plotly.graph_objects as go
import argparse
import os

import torch
import numpy as np
import pandas as pd

from metrics import Barrier

def add_trace(plot, x, y, name, color=None):
    plot.add_trace(go.Scatter(x=x, y=y, name=name, line_color=color))

def add_avg_trace(plot, x, y, name, avg_epochs, color=None):
    add_trace(plot, x, make_smooth(y, avg_epochs), name, color=color)

def make_smooth(y, avg_epochs=100):
    ny = list()
    cur_val = y[0]
    for i in range(len(y)):
        cur_val -= y[max(0, i - avg_epochs)] / avg_epochs
        cur_val += y[i] / avg_epochs
        ny.append(cur_val)

    return ny


def add_vertical_line(plot, x, y_st, y_en, name, color=None):
    add_trace(plot, x=[x, x], y=[y_st, y_en], name=name, color=color)


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
        if plot_data[i] in Barrier.values():
            barriers.append((Barrier(plot_data[i]), len(y_data)))
        else:
            y_data.append(plot_data[i])
            

    y = np.array(make_smooth(y_data, avg_epochs))
    x = np.array(range(len(y)))
    add_trace(plot, x, y, name="metric")

    y_min, y_max = np.min(y), np.max(y)

    for barrier, pos in barriers:
        if barrier == Barrier.EPOCH and not show_epochs:
            continue
        if barrier == Barrier.PRUNE and not show_prune_epochs:
            continue

        colors = {Barrier.EPOCH: 'red', Barrier.PRUNE: 'orange'}
        add_vertical_line(plot, pos, y_min, y_max, name=str(barrier), color=colors[barrier])

    return plot


def load_csv(path):
    dataframe = pd.read_csv(path, index_col=False)
    return dataframe.columns, dataframe.to_numpy()

def get_last_log(logdir):
    return max([os.path.join(logdir, d) for d in os.listdir(logdir)], key=os.path.getmtime)


if __name__ == "__main__":
    log_path = get_last_log("logdir")
    columns, data = load_csv(os.path.join(log_path, "plots", "Exploit_iter0_prune1.0.csv"))
    create_reward_plot(np.squeeze(data), avg_epochs=100).show()

    columns, data = load_csv(os.path.join(log_path, "plots", "qerror.csv"))
    create_metric_plot(np.squeeze(data), avg_epochs=1000, show_epochs=True).show()
