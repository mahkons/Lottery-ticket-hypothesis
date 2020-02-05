import plotly as plt
import plotly.graph_objects as go
import argparse

import torch
import numpy as np
from statistics import mean

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--avg', type=int, default=100, required=False)
    return parser 

paths = [
        "100.000",
        "51.200",
        "20.972",
        "13.422",
    ]

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

if __name__ == "__main__":
    args = create_parser().parse_args()
    data = [torch.load("plots/" + path) for path in paths]
    plot = go.Figure()

    for (plot_data, path) in zip(data, paths):
        y = np.array(plot_data)
        x = np.arange(len(y))
        add_avg_trace(plot, x, y, path, args.avg)

    plot.show()
