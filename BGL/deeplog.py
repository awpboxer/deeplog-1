# -*- coding: utf-8 -*-

import argparse
import sys
sys.path.append('../')

from logdeep.models.lstm import deeplog, deeplog2
from logdeep.tools.predict import Predicter
from logdeep.tools.train import Trainer
from logdeep.tools.utils import *
import torch

data_dir =  "../../data/BGL/"
log_file = "BGL_2k.log.txt"
output_dir = "../output/bgl/"

# Config Parameters
options = dict()
options['data_dir'] = output_dir
options['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

# Smaple
options['sample'] = "sliding_window"
options['window_size'] = 20  # if fix_window

# Features
options['sequentials'] = False
options['quantitatives'] = False
options['semantics'] = False
options['parameters'] = True
options['feature_num'] = sum(
    [options['sequentials'], options['quantitatives'], options['semantics'], options['parameters']])

# Model
options['input_size'] = 1
options['hidden_size'] = 64
options['num_layers'] = 2
options['num_classes'] = 1 # 334 # when options['parameters'] is True, then num_classes = 1

# Train
options['batch_size'] = 128 #2048
options['accumulation_step'] = 1
options['batch_size_train'] = 256
options['batch_size_test'] = 256
options["n_epochs_stop"] = 5

options['optimizer'] = 'adam'
options['lr'] = 0.001
options['max_epoch'] = 2
options['lr_step'] = (options['max_epoch'] - 20, options['max_epoch'])
options['lr_decay_ratio'] = 0.1

options['resume_path'] = None
options['model_name'] = "deeplog"
options['save_dir'] = options["data_dir"] + "deeplog/"

# Predict
options['model_path'] = options["save_dir"] + "deeplog_bestloss.pth"
options['num_candidates'] = 9
options["threshold"] = None
options["gaussian_mean"] = -0.0055
options["gaussian_std"] = 0.0633


print("Device:", options['device'])
seed_everything(seed=1234)
Model = deeplog(input_size=options['input_size'],
                hidden_size=options['hidden_size'],
                num_layers=options['num_layers'],
                num_keys=options['num_classes'])

def train():
    trainer = Trainer(Model, options)
    trainer.start_train()
    plot_train_valid_loss(options["save_dir"])


def predict():
    predicter = Predicter(Model, options)
    predicter.predict_unsupervised()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    train_parser = subparsers.add_parser('train')
    train_parser.set_defaults(mode='train')

    predict_parser = subparsers.add_parser('predict')
    predict_parser.set_defaults(mode='predict')
    predict_parser.add_argument('-n','--num_candidates', type=int, default=9, help='num candidates')
    predict_parser.add_argument('-t','--threshold', type=int, default=0, help='threshold')

    args = parser.parse_args(['predict'])
    print("arguments", args)

    if args.mode == 'train':
        train()
    else:
        options["num_candidates"] = args.num_candidates
        options['threshold'] = args.threshold
        predict()
