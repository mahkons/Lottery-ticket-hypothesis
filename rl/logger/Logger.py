import csv
import datetime
import os


class Logger():
    def __init__(self, logdir):
        self.logdir = logdir
        
        assert(os.path.isdir(logdir))
        filename = "log_" + datetime.datetime.now().isoformat()
        self.dir = os.path.join(logdir, filename)
        os.mkdir(self.dir)

        self.params = dict()
        self.plots = dict()

    def update_params(self, params):
        self.params.update(params)

    def add_plot(self, name):
        self.plots[name] = list()

    def add_plot_point(self, name, point):
        self.plots[name].append(point)

    def get_plot(self, name):
        return self.plots[name]

    def save_logs(self):
        self.save_csv()
        self.save_tensorboard()

    def save_csv(self):
        pass

    def save_tensorboard(self):
        pass
