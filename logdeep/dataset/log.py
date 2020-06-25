#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler
from logdeep.dataset.sample0 import sliding_window, session_window


class log_dataset(Dataset):
    def __init__(self, logs, labels, seq=True, quan=False, sem=False, param=False):
        self.seq = seq
        self.quan = quan
        self.sem = sem
        self.param = param
        if self.seq:
            self.Sequentials = logs['Sequentials']
        if self.quan:
            self.Quantitatives = logs['Quantitatives']
        if self.sem:
            self.Semantics = logs['Semantics']
        if self.param:
            self.Parameters = logs['Parameters']
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        log = dict()
        if self.seq:
            log['Sequentials'] = torch.tensor(self.Sequentials[idx],
                                              dtype=torch.float)
        if self.quan:
            log['Quantitatives'] = torch.tensor(self.Quantitatives[idx],
                                                dtype=torch.float)
        if self.sem:
            log['Semantics'] = torch.tensor(self.Semantics[idx],
                                            dtype=torch.float)
        if self.param:
            log['Parameters'] = torch.tensor(self.Parameters[idx], dtype=torch.float)
        return log, self.labels[idx]


if __name__ == '__main__':
    data_dir = '../../data/'
    window_size = 10
    train_logs = sliding_window(data_dir=data_dir,
                             datatype='train',
                             window_size=window_size)
    train_dataset = log_dataset(log=train_logs, seq=True, quan=True)
    print(train_dataset[0])
    print(train_dataset[100])
