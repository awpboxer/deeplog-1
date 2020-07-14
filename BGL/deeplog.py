# -*- coding: utf-8 -*-

import argparse
import sys
sys.path.append('../')

from logdeep.models.lstm import deeplog, deeplog2
from logdeep.tools.predict import Predicter
from logdeep.tools.train import Trainer
from logdeep.tools.utils import *

data_dir =  "../../data/BGL/"
log_file = "BGL_2k.log.txt"
output_dir = "../output/bgl/"

# Config Parameters
options = dict()
options['data_dir'] = output_dir
options['device'] = "cpu"

# Smaple
options['sample'] = "sliding_window"
options['window_size'] = 20  # if fix_window

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
options['num_layers'] = 2
options['num_classes'] = 334

# Train
options['batch_size'] = 128 #2048
options['accumulation_step'] = 1
options['batch_size_train'] = 256
options['batch_size_test'] = 256
options["n_epochs_stop"] = 5

options['optimizer'] = 'adam'
options['lr'] = 0.001
options['max_epoch'] = 50
options['lr_step'] = (options['max_epoch'] - 20, options['max_epoch'])
options['lr_decay_ratio'] = 0.1

options['resume_path'] = None
options['model_name'] = "deeplog"
options['save_dir'] = options["data_dir"] + "deeplog/"

# Predict
options['model_path'] = options["save_dir"] + "deeplog_bestloss.pth"
options['num_candidates'] = 9
options["threshold"] = 0

seed_everything(seed=1234)



def train():
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
    trainer = Trainer(Model, options)
    trainer.start_train()
    plot_train_valid_loss(options["save_dir"])


def predict():
    if options['parameters']:
        Model = deeplog2(input_size=options['input_size'],
                         hidden_size=options['hidden_size'],
                         num_layers=options['num_layers'],
                         num_keys=options['num_classes'])
        predicter = Predicter(Model, options)
        predicter.predict_unsupervised_with_params()

    else:
        Model = deeplog(input_size=options['input_size'],
                        hidden_size=options['hidden_size'],
                        num_layers=options['num_layers'],
                        num_keys=options['num_classes'])
        predicter = Predicter(Model, options)
        predicter.predict_unsupervised()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    train_parser = subparsers.add_parser('train')
    train_parser.set_defaults(mode='train')
    train_parser.add_argument('-w', type=int, default=10, help="window size")
    train_parser.add_argument('-n', type=int, default=334, help="num classes")
    train_parser.add_argument('-b', type=int, default=32, help="batch size")
    train_parser.add_argument('-s', type=int, default=5, help="num of epoch for early stopping")
    train_parser.add_argument('-e', type=int, default=50, help="max epoch")

    predict_parser = subparsers.add_parser('predict')
    predict_parser.set_defaults(mode='predict')
    predict_parser.add_argument('-n','--num_candidates', type=int, default=9, help='num candidates')
    predict_parser.add_argument('-t','--threshold', type=int, default=0, help='threshold')
    args = parser.parse_args('predict'.split())
    print("arguments", args)
    if args.mode == 'train':
        # options['window_size'] = args.w
        # options['num_classes'] = args.n
        # options['batch_size_train'] = args.b
        # options['batch_size_test'] = 2*args.b
        # options["n_epochs_stop"] = args.s
        # options['max_epoch'] = args.e
        train()
    else:
        # options = dict()
        # with open(output_dir + 'deeplog/parameters.txt') as f:
        #     for line in f.readlines():
        #         k, v = line.strip().split(':')
        #         options[k] = repr(v)
        # options["num_candidates"] = args.num_candidates
        # options['threshold'] = args.threshold
        predict()
