import plotly as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import argparse
import os
import re
import math

import torch
import numpy as np
import pandas as pd
import json

from metrics import Barrier
from metrics.StabilityPlots import detrend, drawdown, variance, IQR, MAD, CVaR, window_percentile, window_mean_among_big

def add_trace(plot, x, y, name, color=None):
    plot.add_trace(go.Scatter(x=x, y=y, name=name, line_color=color))

def add_avg_trace(plot, x, y, name, avg_epochs, color=None):
    add_trace(plot, x, make_smooth(y, avg_epochs), name, color=color)

def make_smooth(y, avg_epochs=1):
    ny = list()
    
    cur_count = 0
    cur_val = 0
    for i in range(len(y)):
        cur_count = min(cur_count + 1, avg_epochs)
        if i >= avg_epochs:
            cur_val -= y[i - avg_epochs]
        cur_val += y[i]

        ny.append(cur_val / cur_count)

    return ny


def add_vertical_line(plot, x, y_st, y_en, name, color=None):
    add_trace(plot, x=[x, x], y=[y_st, y_en], name=name, color=color)


def add_reward_trace(plot, plot_data, use_steps=True, avg_epochs=1, name="reward"):
    if use_steps:
        plot_data = plot_data[np.argsort(plot_data[:, 1])]
    else:
        plot_data = plot_data[np.argsort(plot_data[:, 0])]
    train_episodes, steps, rewards = zip(*plot_data)

    y = np.array(rewards)
    if use_steps:
        x = np.array(steps)
    else:
        x = np.array(train_episodes)
    add_avg_trace(plot, x, y, name=name, avg_epochs=avg_epochs)


def create_reward_plot(plot_data, title="reward plot", use_steps=False, avg_epochs=1):
    plot = go.Figure()
    plot.update_layout(title=title)

    add_reward_trace(plot, plot_data, use_steps, avg_epochs)

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
    return dataframe.to_numpy()

def get_last_log(logdir):
    return max([os.path.join(logdir, d) for d in os.listdir(logdir)], key=os.path.getmtime)


def get_paths(dir, prefix=""):
    return sorted(filter(lambda path: path.startswith(prefix), os.listdir(dir)),
            key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])


def load_data(logpath, plotpath, repeat=None):
    if repeat is None:
        return load_csv(os.path.join(logpath, plotpath))

    paths = [os.path.join("logdir", d) for d in os.listdir("logdir")]
    paths = list(filter(lambda x: x.startswith(logpath), paths))


    if len(paths) != repeat:
        print("Warning: not all repeats! {}".format(paths))

    data_list = list()
    for path in paths:
        data_list.append(load_csv(os.path.join(path, plotpath)))

    data = np.concatenate(data_list)
    return data


def create_distributions_plots(logpath, repeat=None):
    assert repeat is None

    paths = get_paths(os.path.join(logpath, "plots"), "weights")
    assert(len(paths) == 1)
    path = paths[0]


    data = load_data(logpath, os.path.join("plots", path), repeat).squeeze(1)

    sz = len(data)
    n = int(math.sqrt(sz - 1e-9)) + 1
    plot = make_subplots(rows=n, cols=n, start_cell="top-left") 

    pos = 0
    for sweights in data:
        if sweights in Barrier.values():
            continue

        weights = json.loads(sweights)
        print(len(weights))
        plot.add_trace(go.Histogram(x=weights), row=pos//n + 1, col=pos%n + 1)
        pos += 1
        
    return plot

def add_rewards(plot, logpath, use_steps=True, repeat=None, avg_constant=20, transform=lambda x: x):
    if repeat is None:
        dirpath = logpath
    else:
        paths = [os.path.join("logdir", d) for d in os.listdir("logdir")]
        paths = list(filter(lambda x: x.startswith(logpath), paths))
        dirpath = paths[0]

    paths = get_paths(os.path.join(dirpath, "plots"), "Exploit")
    if not paths:
        paths = get_paths(os.path.join(dirpath, "plots"))

    for path in paths:
        data = load_data(logpath, os.path.join("plots", path), repeat)
        data[:, 2] = transform(data[:, 2])
        add_reward_trace(plot, data, use_steps=use_steps, avg_epochs=len(data)//avg_constant + 1, name=logpath + path)

    return plot


def remove_repeat_suffix(x):
    pattern = re.compile("repeat_\d*")
    pos = pattern.search(x)
    if pos != None:
        x = x[:pos.span()[0]]
    return x


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logpath', type=str, default=None, required=False)
    parser.add_argument('--repeat', type=int, default=None, required=False)
    return parser


if __name__ == "__main__":
    args = create_parser().parse_args() 
    if args.logpath is None:
        logpaths = [get_last_log("logdir")]
    else:
        logpaths = list(filter(lambda x: re.fullmatch(args.logpath, x), os.listdir("logdir"))) 
        logpaths = [os.path.join("logdir", d) for d in logpaths]

    if args.repeat:
        logpaths = list(set(map(remove_repeat_suffix, logpaths)))

    
    rewards_plot = go.Figure()
    rewards_plot.update_layout(title="Rewards", xaxis_title="Optimization step", yaxis_title="Reward")

    transform = lambda data: data
    avg_constant = 20
    for logpath in logpaths:
        add_rewards(rewards_plot, logpath, use_steps=True, repeat=args.repeat, avg_constant=avg_constant, transform=transform)

    plt.offline.plot(rewards_plot, filename="generated/rewards_plot.html")

    #  data = load_csv(os.path.join(logpath, "plots", "qerror.csv"))
    #  create_metric_plot(np.squeeze(data), avg_epochs=10000).show()

    #  data = load_csv(os.path.join(logpath, "plots", "stability.csv"))
    #  create_metric_plot(np.squeeze(data), avg_epochs=1).show()

    #  weights_plot = create_distributions_plots(logpath, repeat=args.repeat)
    #  plt.offline.plot(weights_plot, filename="generated/weights_plot.html")
