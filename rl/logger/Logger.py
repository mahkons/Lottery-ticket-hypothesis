import torch
import datetime
import os


class Logger():
    def __init__(self, logdir):
        self.logdir = logdir
        
        assert(os.path.isdir(logdir))
        filename = "log_" + datetime.datetime.now().isoformat()
        os.mkdir(os.path.join(logdir, ""))

        self.params = dict()

    def update_params(self, params):
        self.params.update(params)
