import pandas as pd
import os
import numpy as np
from tensorboardX import SummaryWriter

from metrics import Barrier

logger__ = None

def init_logger(logdir, logname):
    global logger__
    logger__ = Logger(logdir, logname)

def log():
    global logger__
    return logger__

class Logger():
    def __init__(self, logdir, logname):
        self.logdir = logdir
        
        assert(os.path.isdir(logdir))
        self.dir = os.path.join(logdir, logname)
        os.mkdir(self.dir)

        self.tensorboard_dir = os.path.join(self.dir, "tensorboard")
        os.mkdir(self.tensorboard_dir)
        self.tensorboard_writer = SummaryWriter(self.tensorboard_dir)

        self.params = dict()
        self.plots = dict()
        self.plots_columns = dict()

    def update_params(self, params):
        self.params.update(params)

    def add_plot(self, name, columns):
        assert name not in self.plots
        self.plots[name] = list()
        self.plots_columns[name] = columns

    def add_plot_point(self, name, point):
        self.plots[name].append(point)

    def get_plot(self, name):
        return self.plots[name]

    def save_logs(self):
        self.save_csv()
        self.save_tensorboard()

    def save_csv(self):
        plot_path = os.path.join(self.dir, "plots")
        os.mkdir(plot_path)
        for plot_name, plot_data in self.plots.items():
            filename = os.path.join(plot_path, plot_name + ".csv")
            pd.DataFrame(plot_data, columns=self.plots_columns[plot_name]).to_csv(filename, index=False)

        params_path = os.path.join(self.dir, "params.csv")
        pd.DataFrame(self.params.items(), columns=("name", "value")).to_csv(params_path, index=False)

    def save_tensorboard(self):
        self.tensorboard_writer.add_hparams(self.params, {})
        for plot_name, plot_data in self.plots.items():
            for i in range(len(plot_data)):
                # skip barriers
                if plot_data[i] in Barrier.values():
                    continue

                # TODO fix ugly ifs
                if isinstance(plot_data[i], tuple):
                    self.tensorboard_writer.add_scalar(plot_name, plot_data[i][2], i)
                else:
                    self.tensorboard_writer.add_scalar(plot_name, plot_data[i], i)

        self.tensorboard_writer.close()
