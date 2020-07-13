#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gc
import os
import sys
import time
from collections import Counter
sys.path.append('../../')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from logdeep.dataset.log import log_dataset
from logdeep.dataset.sample import session_window, sliding_window
from logdeep.tools.utils import (save_parameters, seed_everything,
                                 train_val_split)


def generate(data_dir, name):
    print("Loading", data_dir + name)
    window_size = 10
    hdfs = {}
    length = 0

    with open(data_dir + name, 'r') as f:
        for ln in f.readlines():
            ln = list(map(eval, ln.strip().strip('"').split()))
            if isinstance(ln[0], int):
                ln = list(map(lambda n: n - 1, ln))
            else:
                params = tuple(map(lambda x: x[1], ln))
                ln = list(map(lambda x: x[0]-1, ln))

            ln = ln + [-1] * (window_size + 1 - len(ln))
            hdfs[tuple(ln)] = hdfs.get(tuple(ln), 0) + 1
            length += 1
    print('Number of session after removing duplicates ({}): {}'.format(name, len(hdfs)))
    return hdfs, length


class Predicter():
    def __init__(self, model, options):
        self.data_dir = options['data_dir']
        self.device = options['device']
        self.model = model
        self.model_path = options['model_path']
        self.window_size = options['window_size']
        self.num_candidates = options['num_candidates']
        self.num_classes = options['num_classes']
        self.input_size = options['input_size']
        self.sequentials = options['sequentials']
        self.quantitatives = options['quantitatives']
        self.semantics = options['semantics']
        self.parameters = options['parameters']
        self.batch_size = options['batch_size']
        self.num_classes = options['num_classes']
        self.threshold = options["threshold"]
        self.batch_size_test = options["batch_size_test"]

    def predict_unsupervised(self):
        model = self.model.to(self.device)
        model.load_state_dict(torch.load(self.model_path)['state_dict'])
        model.eval()
        print('model_path: {}'.format(self.model_path))
        test_normal_loader, test_normal_length = generate(self.data_dir, 'test_normal')
        test_abnormal_loader, test_abnormal_length = generate( self.data_dir, 'test_abnormal')
        print("testing normal size {}, testing abnormal size{}".format(test_normal_length, test_abnormal_length))

        TP = 0
        FP = 0
        # Test the model
        start_time = time.time()
        tt_abnomal_cnt = 0
        with torch.no_grad():
            for idx, line in tqdm(enumerate(test_normal_loader.keys())):
                abnormal_cnt = 0
                for i in range(len(line) - self.window_size):
                    seq0 = line[i:i + self.window_size]
                    label = line[i + self.window_size]
                    seq1 = [0] * self.num_classes
                    log_conuter = Counter(seq0)
                    for key in log_conuter:
                        seq1[key] = log_conuter[key]

                    seq0 = torch.tensor(seq0, dtype=torch.float).view(
                        -1, self.window_size, self.input_size).to(self.device)
                    seq1 = torch.tensor(seq1, dtype=torch.float).view(
                        -1, self.num_classes, self.input_size).to(self.device)
                    label = torch.tensor(label).view(-1).to(self.device)
                    output = model(features=[seq0, seq1], device=self.device)
                    predicted = torch.argsort(output,
                                              1)[0][-self.num_candidates:]
                    if label not in predicted:
                        abnormal_cnt += test_normal_loader[line]
                    if abnormal_cnt > self.threshold:
                        FP += 1
                        break
                tt_abnomal_cnt += abnormal_cnt
                print("test normal, line {}, abnormal cnt {}".format(idx, abnormal_cnt))
            print("test normal, average abnormal cnt {}".format(tt_abnomal_cnt/(idx+1)))

        tt_abnomal_cnt = 0
        with torch.no_grad():
            for idx, line in tqdm(enumerate(test_abnormal_loader.keys())):
                abnormal_cnt = 0
                for i in range(len(line) - self.window_size):
                    seq0 = line[i:i + self.window_size]
                    label = line[i + self.window_size]
                    print(seq0, label)
                    seq1 = [0] * self.num_classes
                    log_conuter = Counter(seq0)
                    for key in log_conuter:
                        seq1[key] = log_conuter[key]

                    seq0 = torch.tensor(seq0, dtype=torch.float).view(
                        -1, self.window_size, self.input_size).to(self.device)
                    seq1 = torch.tensor(seq1, dtype=torch.float).view(
                        -1, self.num_classes, self.input_size).to(self.device)
                    label = torch.tensor(label).view(-1).to(self.device)
                    output = model(features=[seq0, seq1], device=self.device)
                    predicted = torch.argsort(output,
                                              1)[0][-self.num_candidates:]
                    if label not in predicted:
                        abnormal_cnt += test_abnormal_loader[line]
                    if abnormal_cnt > self.threshold: # or (len(line) - self.window_size)*self.threshold it defines fault tolarent rate for each sequence
                        TP += 1
                        break
                tt_abnomal_cnt += abnormal_cnt
                print("test abnormal, line {}, abnormal cnt {}".format(idx, abnormal_cnt))
            print("testab normal, average abnormal cnt {}".format(tt_abnomal_cnt/(idx+1)))

        # Compute precision, recall and F1-measure
        FN = test_abnormal_length - TP
        P = 100 * TP / (TP + FP +1 )
        R = 100 * TP / (TP + FN+1)
        F1 = 2 * P * R / (P + R+1)
        print("TP, FP", TP, FP)
        print(
            'false positive (FP): {}, false negative (FN): {}, Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%'
            .format(FP, FN, P, R, F1))
        print('Finished Predicting')
        elapsed_time = time.time() - start_time
        print('elapsed_time: {}'.format(elapsed_time))


    def predict_unsupervised_with_params(self):
        model = self.model.to(self.device)
        model.load_state_dict(torch.load(self.model_path)['state_dict'])
        model.eval()
        print('model_path: {}'.format(self.model_path))
        test_normal_logs, test_normal_labels = sliding_window(
            data_dir=self.data_dir,
            datatype='test_normal',
            window_size= self.window_size,
            num_classes=self.num_classes,
            sample_ratio=1
        )
        test_abnormal_logs, test_abnormal_labels = sliding_window(
            data_dir=self.data_dir,
            datatype='test_abnormal',
            window_size=self.window_size,
            num_classes=self.num_classes
        )

        test_normal_loader = log_dataset(logs=test_normal_logs,
                                    labels=test_normal_labels,
                                    seq=self.sequentials,
                                    quan=self.quantitatives,
                                    sem=self.semantics,
                                    param=self.parameters)
        test_abnormal_loader = log_dataset(logs=test_abnormal_logs,
                                    labels=test_abnormal_labels,
                                    seq=self.sequentials,
                                    quan=self.quantitatives,
                                    sem=self.semantics,
                                    param=self.parameters)

        TP = 0
        FP = 0
        start_time = time.time()
        normal_tbar = tqdm(test_normal_loader)
        with torch.no_grad():
            for i, (log, label) in enumerate(normal_tbar):
                    features = []
                    for value in log.values():
                        features.append(value.clone().view(-1, self.window_size, self.input_size).to(self.device))
                    output = model(features=features, device=self.device)
                    predicted = torch.argsort(output,
                                            1)[0][-self.num_candidates:]
                    if label not in predicted:
                        FP += 1
                        break

        abnormal_tbar = tqdm(test_abnormal_loader)
        with torch.no_grad():
            for i, (log, label) in enumerate(abnormal_tbar):
                features = []
                for value in log.values():
                    features.append(value.clone().view(-1,self.window_size, self.input_size).to(self.device))
                output = model(features=features, device=self.device)
                predicted = torch.argsort(output,
                                              1)[0][-self.num_candidates:]
                if label not in predicted:
                    TP += 1
                    break

        # Compute precision, recall and F1-measure
        FN = len(test_abnormal_labels) - TP
        P = 100 * TP / (TP + FP +1)
        R = 100 * TP / (TP + FN +1)
        F1 = 2 * P * R / (P + R +1)
        print(
            'false positive (FP): {}, false negative (FN): {}, Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%'
            .format(FP, FN, P, R, F1))
        print('Finished Predicting')
        elapsed_time = time.time() - start_time
        print('elapsed_time: {}'.format(elapsed_time))



    def predict_supervised(self):
        model = self.model.to(self.device)
        model.load_state_dict(torch.load(self.model_path)['state_dict'])
        model.eval()
        print('model_path: {}'.format(self.model_path))
        test_logs, test_labels = session_window(self.data_dir, datatype='test')
        test_dataset = log_dataset(logs=test_logs,
                                   labels=test_labels,
                                   seq=self.sequentials,
                                   quan=self.quantitatives,
                                   sem=self.semantics)
        self.test_loader = DataLoader(test_dataset,
                                      batch_size=self.batch_size_test,
                                      shuffle=False,
                                      pin_memory=True)
        tbar = tqdm(self.test_loader, desc="\r")
        TP, FP, FN, TN = 0, 0, 0, 0
        for i, (log, label) in enumerate(tbar):
            features = []
            for value in log.values():
                features.append(value.clone().to(self.device))
            output = self.model(features=features, device=self.device)
            output = F.sigmoid(output)[:, 0].cpu().detach().numpy()
            # predicted = torch.argmax(output, dim=1).cpu().numpy()
            predicted = (output < 0.2).astype(int)
            label = np.array([y.cpu() for y in label])
            TP += ((predicted == 1) * (label == 1)).sum()
            FP += ((predicted == 1) * (label == 0)).sum()
            FN += ((predicted == 0) * (label == 1)).sum()
            TN += ((predicted == 0) * (label == 0)).sum()
        P = 100 * TP / (TP + FP)
        R = 100 * TP / (TP + FN)
        F1 = 2 * P * R / (P + R)
        print(
            'false positive (FP): {}, false negative (FN): {}, Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%'
            .format(FP, FN, P, R, F1))
