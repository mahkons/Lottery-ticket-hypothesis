class Config():
    def __init__(self):
        pass

    def to_dict(self):
        res_dict = dict()
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                res_dict.update(value.to_dict())
            else:
                res_dict[key] = value
        return res_dict


class OptimizerConfig(Config):
    def __init__(self):
        pass


class MemoryConfig(Config):
    def __init__(self):
        pass
