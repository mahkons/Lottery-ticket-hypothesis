class CommonRescale():
    def get_factor(self, a):
        raise NotImplementedError

    def __call__(self, init, mask):
        init_copy = init.clone()
        init_copy[~mask] = 1e-12
        return self.get_factor(init) / self.get_factor(init_copy)


class L1GlobalRescale(CommonRescale):
    def get_factor(self, a):
        return a.abs().sum()


class L2GlobalRescale(CommonRescale):
    def get_factor(self, a):
        return a.norm()


class L1LocalRescale(CommonRescale):
    def get_factor(self, a):
        if len(a.shape) == 1:
            return a.abs()
        if len(a.shape) == 2: # Linear
            return a.abs().sum(dim=1, keepdims=True)
        if len(a.shape) == 4: # Conv2d
            return a.abs().sum(dim=(1, 2, 3), keepdims=True)


class L2LocalRescale(CommonRescale):
    def get_factor(self, a):
        if len(a.shape) == 1: # Bias
            return a.abs()
        if len(a.shape) == 2: # Linear
            return a.norm(dim=1).unsqueeze(1)
        if len(a.shape) == 4: # Conv2d
            return a.reshape(a.shape[0], -1).norm(dim=1)[(..., None, None, None)]
