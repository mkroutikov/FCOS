# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from collections import defaultdict
from collections import deque

import torch


class EmaValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20):
        self._lam = 1. / window_size
        self.ema = None
        self.last = None

    def update(self, value):
        if self.ema is None:
            self.ema = value  # "fast" start
        else:
            self.ema += (value - self.ema) * self._lam
        self.last = value

    def __repr__(self):
        if self.ema is None:
            return 'None'
        return '%.4f(%.4f)' % (self.ema, self.last)


class MetricLogger(object):
    def __init__(self, delimiter=", "):
        self.meters = defaultdict(EmaValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
                    type(self).__name__, attr))

    def __str__(self):
        return self.delimiter.join(
            '%s: %r' % (name, val)
            for name, val in self.meters.items()
        )
