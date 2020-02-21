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
        plot_path = os.path.join(self.dir, "plots")
        os.mkdir(plot_path)
        for plot_name, plot_data in self.plots.items():
            with open(os.path.join(plot_path, plot_name + ".csv"), "w+", newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(("train_episode", "train_steps", "reward"))
                writer.writerows(plot_data)


        params_path = os.path.join(self.dir, "params.csv")
        with open(params_path, "w+", newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(("name", "value"))
            writer.writerows(self.params.items())


    def save_tensorboard(self):
        pass
