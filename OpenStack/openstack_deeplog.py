#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import sys
sys.path.append('../')

from logdeep.models.lstm import deeplog, loganomaly, robustlog, deeplog2
from logdeep.tools.predict import Predicter
from logdeep.tools.train import Trainer
from logdeep.tools.utils import *
from OpenStack import const, data_view

# Config Parameters

options = dict()
options['data_dir'] = const.OUTPUT_DIR + const.PARSER + "_result2/"
options['device'] = "cpu"

# Smaple
options['sample'] = "sliding_window"
options['window_size'] = 6  # if fix_window

# Features
options['sequentials'] = True
options['quantitatives'] = False
options['semantics'] = False
options['parameters'] = False
options['feature_num'] = sum(
    [options['sequentials'], options['quantitatives'], options['semantics'], options['parameters']])

# Model
options['input_size'] = 1
options['hidden_size'] = 64
options['num_layers'] = 3
options['num_classes'] = 49

# Train
options['batch_size'] = 128 #2048
options['accumulation_step'] = 1
options['batch_size_train'] = 64
options['batch_size_test'] = 1024
options["n_epochs_stop"] = 5

options['optimizer'] = 'adam'
options['lr'] = 0.001
options['max_epoch'] = 50
options['lr_step'] = (40, 50)
options['lr_decay_ratio'] = 0.1

options['resume_path'] = None
options['model_name'] = "deeplog"
options['save_dir'] = options["data_dir"] + "deeplog/"

# Predict
options['model_path'] = options["save_dir"] + "deeplog_bestloss.pth"
options['num_candidates'] = 9
options["threshold"] = 0

seed_everything(seed=1234)

if options['parameters']:
    Model = deeplog2(input_size=options['input_size'],
                     hidden_size=options['hidden_size'],
                     num_layers=options['num_layers'],
                     num_keys=options['num_classes'])
else:
    Model = deeplog(input_size=options['input_size'],
                    hidden_size=options['hidden_size'],
                    num_layers=options['num_layers'],
                    num_keys=options['num_classes'])

def train():
    trainer = Trainer(Model, options)
    trainer.start_train()
    data_view.plot_train_valid_loss(options["save_dir"])

def predict():
    predicter = Predicter(Model, options)
    #predicter.predict_unsupervised_with_params()
    if options['parameters']:
        predicter.predict_unsupervised_with_params()
    else:
        predicter.predict_unsupervised()


if __name__ == "__main__":
    #train()
    #predict()
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train', 'predict'])
    args = parser.parse_args()
    if args.mode == 'train':
        train()
    else:
        predict()
