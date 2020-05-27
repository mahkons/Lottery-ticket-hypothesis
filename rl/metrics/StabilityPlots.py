import numpy as np
from collections import deque

def detrend(xs):
    return np.insert(xs[1:] - xs[:-1], 0, 0)

def drawdown(xs):
    cur_max = 0
    ys = list()
    for x in xs:
        cur_max = max(cur_max, x)
        ys.append(x - cur_max)
    return ys

def variance(xs, window_size):
    window_sum = 0
    window_square_sum = 0
    dlist = list()

    for i in range(len(xs)):
        if i >= window_size:
            window_sum -= xs[i - window_size]
            window_square_sum -= xs[i - window_size] ** 2
        window_sum += xs[i]
        window_square_sum += xs[i] ** 2

        length = min(i + 1, window_size)
        dispersion = window_square_sum / length  - (window_sum / length) ** 2
        dlist.append(dispersion)

    return dlist

def IQR(xs, window_size):
    ilist = list()
    window = deque()

    for i in range(len(xs)):
        if i >= window_size:
            window.popleft()
        window.append(xs[i])

        a = np.array(window)
        iqr = np.percentile(a, 75) - np.percentile(a, 25)
        ilist.append(iqr)

    return np.array(ilist)


def window_percentile(xs, percentile, window_size):
    plist = list()
    window = deque()

    for i in range(len(xs)):
        if i >= window_size:
            window.popleft()
        window.append(xs[i])

        a = np.array(window)
        p = np.percentile(a, percentile)
        plist.append(p)

    return np.array(plist)


def window_mean_among_big(xs, percentile, window_size):
    mlist = list()
    window = deque()

    for i in range(len(xs)):
        if i >= window_size:
            window.popleft()
        window.append(xs[i])

        a = np.array(window)
        p = np.percentile(a, percentile)
        a = a[a >= p]
        if a.size:
            mlist.append(a[a >= p].mean())

    return np.array(mlist)


# Median absolute deviation
def MAD(xs, window_size):
    mlist = list()
    window = deque()

    for i in range(len(xs)):
        if i >= window_size:
            window.popleft()
        window.append(xs[i])

        a = np.array(window)
        median = np.median(np.abs(a - np.mean(a)))
        mlist.append(median)

    return np.array(mlist)


# Expected value in the leftmost tail of distribution (Conditional Value at Risk)
def CVaR(xs, window_size, quantile):
    clist = list()
    window = deque()

    for i in range(len(xs)):
        if i >= window_size:
            window.popleft()
        window.append(xs[i])

        a = np.array(window)
        cvar = np.mean(a[a <= np.quantile(a, quantile)])
        clist.append(cvar)

    return np.array(clist)
