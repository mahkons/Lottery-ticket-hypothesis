from enum import IntEnum

# ugly but works. and saved as numbers to csv
# TODO fix?
class Barrier(IntEnum):
    EPOCH = -123456789
    PRUNE = -123456789 + 1

    @classmethod
    def values(cls):
        return list(map(int, cls))
