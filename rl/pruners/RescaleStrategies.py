class L1GlobalRescale():
    def __call__(self, init, mask):
        return init.abs().sum() / init[~mask].abs().sum()


class L1LocalRescale():
    def __call__(self, init, mask):
        pass


class L2LocalRescale():
    def __call__(self, init, mask):
        pass
