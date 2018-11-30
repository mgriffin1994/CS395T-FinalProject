# -*- coding: utf-8 -*-
"""logger.py

Loggers

"""
import json
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='')

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, dim=1):
        self.dim = dim
        self.reset()

    def reset(self):
        if self.dim == 1:
            self.val = 0
            self.avg = 0
            self.sum = 0
        else:
            self.val = np.zeros(dim)
            self.avg = np.zeros(dim)
            self.sum = np.zeros(dim)
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Logger(object):
    """Training process logger

    Note:
        Used by BaseTrainer to save training history.
    """
    def __init__(self):
        self.entries = {}

    def add_entry(self, entry):
        self.entries[len(self.entries) + 1] = entry

    def __str__(self):
        return json.dumps(self.entries, sort_keys=True, indent=4)
