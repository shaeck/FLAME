#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
import random
from torch import nn


def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

def BadFedAvg(w):
    w_avg_bad = copy.deepcopy(w[0])
    # random number between 1 and len(w) to leave out
    # rand_key = random.choice(list(w_avg.keys()))
    # w_avg[rand_key] += torch.tensor(0.1, dtype=w_avg[rand_key].dtype)

    for k in w_avg_bad.keys():
        w_avg_bad[k] += torch.tensor(0.01, dtype=w_avg_bad[k].dtype)
    for k in w_avg_bad.keys():
        for i in range(1, len(w)):
            w_avg_bad[k] += w[i][k]
        w_avg_bad[k] = torch.div(w_avg_bad[k], len(w))
    return w_avg_bad