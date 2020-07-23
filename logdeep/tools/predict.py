#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gc
import os
import sys
import time
from collections import Counter, defaultdict
sys.path.append('../../')

import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
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


def generate(data_dir, name, window_size):
    print("Loading", data_dir + name)
    dd = {}
    length = 0

    with open(data_dir + name, 'r') as f:
        for ln in f.readlines():
            ln = list(map(eval, ln.strip().split()))
            # log key + params
            if isinstance(ln[0], tuple):
                ln = ln + [(0,0) for _ in range(window_size + 1 - len(ln))]
            else:
                ln = ln + [0] * (window_size + 1 - len(ln))
            dd[tuple(ln)] = dd.get(tuple(ln), 0) + 1
            length += 1
    print('Number of session after removing duplicates ({}): {}'.format(name, len(dd)))
    return dd, length


def detect_logkey_anomaly(output, label, num_candidates):
        predicted = torch.argsort(output, 1)[0][-num_candidates:]
        if label not in predicted:
            return True
        return False


def detect_params_anomaly(output, label, gaussian_mean, gaussian_std):
    predicted = output
    error = predicted.item() - label.item()
    if error < gaussian_mean - 2 * gaussian_std or error > gaussian_mean + 2 * gaussian_std:
        return True
    return False


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
        self.gaussian_mean = options["gaussian_mean"]
        self.gaussian_std = options["gaussian_std"]
        self.save_dir = options['save_dir']

    def predict_unsupervised(self):
        model = self.model.to(self.device)
        model.load_state_dict(torch.load(self.model_path)['state_dict'])
        model.eval()
        print('model_path: {}'.format(self.model_path))
        test_normal_loader, test_normal_length = generate(self.data_dir, 'test_normal', self.window_size)
        test_abnormal_loader, test_abnormal_length = generate( self.data_dir, 'test_abnormal', self.window_size)
        print("testing normal size: {}, testing abnormal size: {}".format(test_normal_length, test_abnormal_length))

        TP = 0
        FP = 0
        # Test the model
        start_time = time.time()
        tt_normal_cnt = defaultdict(int)
        normal_errors = []
        with torch.no_grad():
            for idx, line in tqdm(enumerate(test_normal_loader.keys())):
                abnormal_cnt = 0
                for i in range(len(line) - self.window_size):
                    seq0 = line[i:i + self.window_size]
                    label = line[i + self.window_size]
                    seq0 = torch.tensor(seq0, dtype=torch.float).view(
                        -1, self.window_size, self.input_size).to(self.device)

                    label = torch.tensor(label).view(-1).to(self.device)
                    output = model(features=[seq0], device=self.device)

                    is_anomaly = False
                    if self.sequentials:
                        is_anomaly = detect_logkey_anomaly(output, label, self.num_candidates)
                    if not is_anomaly and self.parameters:
                        normal_errors.append((output[0] - label).item())  ########
                        is_anomaly = detect_params_anomaly(output, label, self.gaussian_mean, self.gaussian_std)
                    if is_anomaly:
                        abnormal_cnt += test_normal_loader[line]

                tt_normal_cnt[abnormal_cnt] += 1
            print("(test normal), abnormal cnt : {}".format(tt_normal_cnt))

        tt_abnormal_cnt = defaultdict(int)
        abnormal_errors = []
        with torch.no_grad():
            for idx, line in tqdm(enumerate(test_abnormal_loader.keys())):
                abnormal_cnt = 0
                for i in range(len(line) - self.window_size):
                    seq0 = line[i:i + self.window_size]
                    label = line[i + self.window_size]

                    seq0 = torch.tensor(seq0, dtype=torch.float).view(
                        -1, self.window_size, self.input_size).to(self.device)
                    label = torch.tensor(label).view(-1).to(self.device)
                    output = model(features=[seq0], device=self.device)

                    is_anomaly = False
                    if self.sequentials:
                        is_anomaly = detect_logkey_anomaly(output, label, self.num_candidates)
                    if not is_anomaly and self.parameters:
                        abnormal_errors.append(output[0].item() - label.item())
                        is_anomaly = detect_params_anomaly(output, label, self.gaussian_mean, self.gaussian_std)
                    if is_anomaly:
                        abnormal_cnt += test_abnormal_loader[line]
                tt_abnormal_cnt[abnormal_cnt] += 1
            print("(test abnormal), abnormal cnt: {}".format(tt_abnormal_cnt))

        res = [0, 0, 0, 0, 0, 0, 0, 0] # th,tp, tn, fp, fn,  p, r, f1
        for th in range(10):
            FP = sum([v for k,v in tt_normal_cnt.items() if k > th])
            TP = sum([v for k,v in tt_abnormal_cnt.items() if k > th])
            # Compute precision, recall and F1-measure
            TN = test_normal_length - FP
            FN = test_abnormal_length - TP
            P = 100 * TP / (TP + FP + 1)
            R = 100 * TP / (TP + FN + 1)
            F1 = 2 * P * R / (P + R + 1)
            if F1 > res[-1]:
                res = [th, TP, TN, FP, FN, P, R, F1]
        TH, TP, TN, FP, FN, P, R, F1 = res
        print('the best threshold', TH)
        print("confusion matrix")
        print("TP: {}, TN: {}, FP: {}, FN: {}".format(TP, TN, FP, FN))
        print(
            'Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%'
            .format(P, R, F1))
        print('Finished Predicting')
        elapsed_time = time.time() - start_time
        print('elapsed_time: {}'.format(elapsed_time))

        if self.parameters:
            sns.distplot(normal_errors, norm_hist=True, label='normal errors')
            sns.distplot(abnormal_errors, norm_hist=True, label='abnormal errors')
            x = np.linspace(self.gaussian_mean - 3 * self.gaussian_std, self.gaussian_mean + 3 * self.gaussian_std, 100)
            plt.plot(x, stats.norm.pdf(x, self.gaussian_mean, self.gaussian_std), label='gaussian')
            plt.legend()
            print("save error distribution")
            plt.savefig(self.save_dir + 'error_distrubtion.png')
            plt.show()

    def predict_unsupervised2(self):
        model = self.model.to(self.device)
        model.load_state_dict(torch.load(self.model_path)['state_dict'])
        model.eval()
        print('model_path: {}'.format(self.model_path))
        test_normal_loader, test_normal_length = generate(self.data_dir, 'test_normal', self.window_size)
        test_abnormal_loader, test_abnormal_length = generate(self.data_dir, 'test_abnormal', self.window_size)
        print("testing normal size: {}, testing abnormal size: {}".format(test_normal_length, test_abnormal_length))

        # Test the model
        start_time = time.time()
        tt_normal_cnt = defaultdict(int)
        normal_errors = []
        with torch.no_grad():
            for idx, line in tqdm(enumerate(test_normal_loader.keys())):
                abnormal_cnt = 0
                params = list(map(lambda x: x[1], line))
                logkeys = list(map(lambda x: x[0], line))
                for i in range(len(line) - self.window_size):
                    seq0 = logkeys[i:i + self.window_size]
                    label0 = logkeys[i + self.window_size]
                    seq1 = params[i:i + self.window_size]
                    label1 = params[i + self.window_size]

                    seq0 = torch.tensor(seq0, dtype=torch.long).view(-1, self.window_size).to(self.device) # add embedding layer
                    label0 = torch.tensor(label0, dtype=torch.long).view(-1).to(self.device)
                    seq1 = torch.tensor(seq1, dtype=torch.float).view(-1, self.window_size, self.input_size).to(self.device)
                    label1 = torch.tensor(label1, dtype=torch.float).view(-1).to(self.device)
                    output0, output1 = model(features=[seq0, seq1], device=self.device)
                    output1 = output1.squeeze()

                    is_anomaly = False
                    if self.sequentials:
                        is_anomaly = detect_logkey_anomaly(output0, label0, self.num_candidates)
                    if not is_anomaly and self.parameters:
                        normal_errors.append(float(output1) - float(label1))
                        is_anomaly = detect_params_anomaly(output1, label1, self.gaussian_mean, self.gaussian_std)
                    if is_anomaly:
                        abnormal_cnt += test_normal_loader[line]

                tt_normal_cnt[abnormal_cnt] += 1
            print("(test normal), abnormal cnt : {}".format(tt_normal_cnt))

        tt_abnormal_cnt = defaultdict(int)
        abnormal_errors = []
        with torch.no_grad():
            for idx, line in tqdm(enumerate(test_abnormal_loader.keys())):
                abnormal_cnt = 0
                params = list(map(lambda x: x[1], line))
                logkeys = list(map(lambda x: x[0], line))
                for i in range(len(line) - self.window_size):
                    seq0 = logkeys[i:i + self.window_size]
                    label0 = logkeys[i + self.window_size]
                    seq1 = params[i:i + self.window_size]
                    label1 = params[i + self.window_size]

                    seq0 = torch.tensor(seq0, dtype=torch.long).view(-1, self.window_size).to(self.device)
                    label0 = torch.tensor(label0).view(-1).to(self.device)
                    seq1 = torch.tensor(seq1, dtype=torch.float).view(-1, self.window_size, self.input_size).to(self.device)
                    label1 = torch.tensor(label1, dtype=torch.float).view(-1).to(self.device)
                    output0, output1 = model(features=[seq0, seq1], device=self.device)
                    output1 = output1.squeeze()

                    is_anomaly = False
                    if self.sequentials:
                        is_anomaly = detect_logkey_anomaly(output0, label0, self.num_candidates)
                    if not is_anomaly and self.parameters:
                        abnormal_errors.append(output1.item() - label1.item())
                        is_anomaly = detect_params_anomaly(output1, label1, self.gaussian_mean, self.gaussian_std)
                    if is_anomaly:
                        abnormal_cnt += test_abnormal_loader[line]

                tt_abnormal_cnt[abnormal_cnt] += 1
            print("(test abnormal), abnormal cnt: {}".format(tt_abnormal_cnt))

        res = [0, 0, 0, 0, 0, 0, 0, 0]  # th,tp, tn, fp, fn,  p, r, f1
        for th in range(10):
            FP = sum([v for k, v in tt_normal_cnt.items() if k > th])
            TP = sum([v for k, v in tt_abnormal_cnt.items() if k > th])
            # Compute precision, recall and F1-measure
            TN = test_normal_length - FP
            FN = test_abnormal_length - TP
            P = 100 * TP / (TP + FP + 1)
            R = 100 * TP / (TP + FN + 1)
            F1 = 2 * P * R / (P + R + 1)
            if F1 > res[-1]:
                res = [th, TP, TN, FP, FN, P, R, F1]
        TH, TP, TN, FP, FN, P, R, F1 = res
        print('the best threshold', TH)
        print("confusion matrix")
        print("TP: {}, TN: {}, FP: {}, FN: {}".format(TP, TN, FP, FN))
        print(
            'Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%'
                .format(P, R, F1))
        print('Finished Predicting')
        elapsed_time = time.time() - start_time
        print('elapsed_time: {}'.format(elapsed_time))

        if self.parameters:
            sns.kdeplot(normal_errors, label='normal errors')
            sns.kdeplot(abnormal_errors, label='abnormal errors')
            x = np.linspace(self.gaussian_mean - 3 * self.gaussian_std, self.gaussian_mean + 3 * self.gaussian_std, 100)
            plt.plot(x, stats.norm.pdf(x, self.gaussian_mean, self.gaussian_std), label='gaussian')
            plt.legend()
            print("save error distribution")
            plt.savefig(self.save_dir + 'error_distrubtion.png')
            plt.show()


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
                                      batch_size=self.batch_size,
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
