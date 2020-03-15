from configs.Config import Config


class Experiment(Config):
    def __init__(
            self,
            opt_steps,
            episodes,
            prune_iters,
            prune_percent,
            device,
            logname,
            random_seed,
            env,
            hyperparams,
            stop_criterion,
            pruner,
        ):
        super(Experiment, self).__init__()
        self.opt_steps = opt_steps
        self.episodes = episodes
        self.prune_iters = prune_iters
        self.prune_percent = prune_percent
        self.device = device 
        self.logname = logname 
        self.random_seed = random_seed 
        self.env = env 
        self.hyperparams = hyperparams 
        self.stop_criterion = stop_criterion 
        self.pruner = pruner 
